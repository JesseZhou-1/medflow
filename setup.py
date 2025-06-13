from setuptools import setup, find_packages

# Safely load long description
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="medflow",
    version="0.1.2",  # bump for each release
    author="Jesse Zhou",
    author_email="jessezhou1@uchicago.edu",
    description=(
        "A Python package for simulation-based causal mediation analysis "
        "with multiple mediators using causal-Graphical Normalizing Flows."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JesseZhou-1/medflow",
    license="BSD-3-Clause",

    packages=find_packages(exclude=["tests*", "venv*"]),
    include_package_data=True,
    zip_safe=False,
    python_requires=">=3.7",

    install_requires=[
        "torch",
        "numpy",
        "pandas",
        "networkx",
        "scikit-learn",
        "causalgraphicalmodels",
        "UMNN",
        "joblib",
        "cGNF @ git+https://github.com/cGNF-Dev/cGNF.git@main#egg=cGNF",
    ],

    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Visualization",
    ],

    project_urls={
        "Documentation": "https://github.com/JesseZhou-1/medflow#readme",
        "Source": "https://github.com/JesseZhou-1/medflow",
        "Tracker": "https://github.com/JesseZhou-1/medflow/issues",
    },
)
