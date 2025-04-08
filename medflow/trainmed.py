import pandas as pd
import numpy as np
import torch  # Importing the PyTorch library, which provides tools for deep learning.
import pickle  # Importing the pickle module, which allows to serialize and deserialize Python object structures.
import ast
from cGNF import train
import collections.abc
collections.Iterable = collections.abc.Iterable
import networkx as nx
from causalgraphicalmodels import CausalGraphicalModel


def train.med(path="", dataset_name="", treatment='', confounder=None, mediator=None, outcome=None,
           test_size=0.2, cat_var=None, sens_corr=None, seed_split=None,
           model_name='models', resume=False,
           trn_batch_size=128, val_batch_size=2048, learning_rate=1e-4, seed=None,
           nb_epoch=50000, nb_estop=50, val_freq=1,
           emb_net=[100, 90, 80, 70, 60],
           int_net=[60, 50, 40, 30, 20]):

    nodes = [treatment, outcome] + mediator + confounder
    edges = []

    # Add edges based on rules
    for c in confounder:
        edges += [(c, n) for n in nodes if n not in confounder]
    edges += [(treatment, m) for m in mediator + [outcome]]
    for i, m in enumerate(mediator):
        edges += [(m, mn) for mn in mediator[i + 1:]]  # Connecting to subsequent mediators
        edges += [(m, outcome)]  # Connecting to outcome

    simDAG = CausalGraphicalModel(nodes=nodes, edges=edges)

    df_cDAG = nx.to_pandas_adjacency(simDAG.dag, dtype=int)  # Converts the DAG to a pandas adjacency matrix.

    # Read the data file.
    df = pd.read_csv(path + dataset_name + '.csv')

    ordered_columns = df.columns.tolist()  # Get the column names from the DataFrame

    df_cDAG = df_cDAG.reindex(index=ordered_columns, columns=ordered_columns)  # Reorder both rows and columns

    df_cDAG.to_csv(path + dataset_name + '_DAG.csv')

    df.dropna(inplace=True)

    print("------- Adjacency Matrix -------")
    print(df_cDAG)

    num_vars = len(df.columns)
    corr_matrix = np.eye(num_vars)

    # If edge strengths are provided, update the strength matrix.
    if sens_corr:
        # Create a mapping of column name to index to use in the matrix.
        col_to_index = {col: idx for idx, col in enumerate(df.columns)}

        # Update the corr_matrix with the specified strengths.
        for key, strength in sens_corr.items():
            try:
                # Check if the key is already a tuple or needs to be converted from string
                if isinstance(key, str):
                    source, target = ast.literal_eval(key)
                elif isinstance(key, tuple):
                    source, target = key
                else:
                    raise ValueError("Key must be a tuple or a string representation of a tuple")

            except ValueError:
                # Handle any errors in conversion
                print(f"Invalid tuple format: {key}")
                continue

            if source in col_to_index and target in col_to_index:
                idx_source = col_to_index[source]
                idx_target = col_to_index[target]

                # Set the corresponding values in the matrix, and its symmetric counterpart.
                corr_matrix[idx_source][idx_target] = strength
                corr_matrix[idx_target][idx_source] = strength

        # Save the updated correlation matrix
        pd.DataFrame(corr_matrix, index=df.columns, columns=df.columns).to_csv(path + 'sens_corr_matrix.csv')

    # Imports the function 'train_test_split' from sklearn's model selection module.
    from sklearn.model_selection import train_test_split
    # Splits the DataFrame 'df' into a training set and a validation set. This function returns two dataframes: the training set and the validation set.
    df_train, df_val = train_test_split(df, test_size=test_size, random_state=seed_split)

    # Converts the training and validation datasets from pandas DataFrame to numpy arrays. These arrays will be used for further data processing.
    df_trn, df_val = df_train.to_numpy(), df_val.to_numpy()

    # Vertically stacks the training and validation arrays into one array. This combined array is used for calculating the mean and standard deviation for data standardization.
    data = np.vstack((df_trn, df_val))

    mu = data.mean(
        axis=0)  # Calculates the mean of 'data' along the column axis. This mean will be used for data standardization.
    sig = data.std(
        axis=0)  # Calculates the standard deviation of 'data' along the column axis. This standard deviation will be used for data standardization.

    all_df_columns = list(df)  # get the list of all the variable names in the dataset
    dict_cat_dims = {}  # Initialize an empty dictionary 'dict_cat_dims' to store the maximum value of each categorical column + 1, which essentially reflects the number of unique categories.

    # Prepare to dequantize
    if cat_var:
        # loop to change each column to category type to get the dimension/position of the categorical variable in the dataframe and respective unique categories.
        dict_unique_cats = {}  # Initialize an empty dictionary 'dict_unique_cats' to store unique categories for each categorical column.

        # The enumerate function is used when you want to iterate over an iterable and also want to have an index attached to each element.
        for i, col in enumerate(
                all_df_columns):  # Loop over each column in the DataFrame. 'i' is the index and 'col' is the column name.
            if col in cat_var:  # Check if the column 'col' is in the list of categorical column names 'cat_col_names'.
                df[col] = df[col].astype('category',
                                         copy=False)  # If so, convert that column to 'category' type. This is often used to save memory or to perform some pandas operations faster.
                dict_unique_cats[col] = list(df[
                                                 col].unique())  # Add the unique categories of the column 'col' to the dictionary 'dict_unique_cats'.
                print(
                    f'\n{col}: {len(dict_unique_cats[col])} categories - value: {dict_unique_cats[col]}')  # Print the index, column name, number of unique categories and the unique categories themselves.
                # dict_cat_dims[i] = len(dict_unique_cats[col])
                dict_cat_dims[i] = max(dict_unique_cats[
                                           col]) + 1  # Instead, it sets the value in 'dict_cat_dims' for the key 'i' to one more than the maximum category of column 'col'. This assumes that the categories are numerical and can be ordered.

    # A dictionary 'pickle_objects' is created to hold the necessary preprocessed variables.
    # These will later be saved into a pickle file for easy reloading in future sessions.
    pickle_objects = {}

    # Adding various data and information to the dictionary.
    pickle_objects['df'] = df  # The entire DataFrame.
    pickle_objects['trn'] = df_trn  # The training data.
    pickle_objects['val'] = df_val  # The validation data.
    pickle_objects['mu'] = mu  # The column-wise mean of the data.
    pickle_objects['sig'] = sig  # The column-wise standard deviation of the data.
    pickle_objects['df_all_columns'] = all_df_columns  # All column names of the DataFrame.
    pickle_objects['df_cat_columns'] = cat_var  # The categorical column names.
    pickle_objects['cat_dims'] = dict_cat_dims  # The dimensions of the categorical columns.
    pickle_objects['seed'] = seed_split  # The random seed.
    pickle_objects['treatment'] = treatment
    pickle_objects['outcome'] = outcome
    pickle_objects['mediator'] = mediator
    pickle_objects['confounder'] = confounder
    pickle_objects['dataset_filepath'] = path + dataset_name  # The file path of the dataset.
    pickle_objects['A'] = torch.from_numpy(
        df_cDAG.to_numpy().transpose()).float()  # The adjacency matrix of the causal graph, converted to a PyTorch tensor.
    pickle_objects['Z_Sigma'] = torch.from_numpy(corr_matrix).float()

    # The context manager 'with open' is used to open a file in write-binary mode ('wb').
    # The pickle.dump function is then used to write the 'pickle_objects' dictionary to this file.
    with open(path + dataset_name + '.pkl', "wb") as f:
        pickle.dump(pickle_objects, f)

    train(path=path, dataset_name=dataset_name, model_name=model_name,
          resume=resume,
          trn_batch_size=trn_batch_size, val_batch_size=val_batch_size, learning_rate=learning_rate, seed=seed,
          nb_epoch=nb_epoch, nb_estop=nb_estop, val_freq=val_freq,
          emb_net=emb_net,
          int_net=int_net)

