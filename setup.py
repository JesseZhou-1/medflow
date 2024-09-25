from setuptools import setup, find_packages

setup(
    name='medsimGNF',
    version='0.9.4',  # start with a small number and increment it with every change
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
        'cGNF'
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

