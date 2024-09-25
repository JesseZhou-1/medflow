from setuptools import setup, find_packages

setup(
    name='medsimGNF',
    version='0.1.0',  # Increment as needed
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'pandas',
        'networkx',
        'scikit-learn',
        'causalgraphicalmodels',
        'UMNN',
        'joblib',
        'cGNF @ git+https://github.com/username/cGNF.git@main#egg=cGNF',
    ],
    author='MedSim-team',
    author_email='medsim.team@gmail.com',
    description='',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url='https://github.com/MedSim-Dev/medsimGNF',
    license='BSD License',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
    ],
)
