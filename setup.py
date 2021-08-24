from setuptools import find_packages, setup

setup(
    name="group_robustness_fairness",
    version="1.0.0",
    description="A description",
    url="https://github.com/som-shahlab/subpopulation_robustness",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas>=1.0.0",
        "torch>=1.6",
        "pyyaml",
        "pyarrow",
        "scipy",
        "sklearn",
        "configargparse",
        "matplotlib",
        "sqlalchemy",
        "dask>=2.14.0",
        "google-cloud-bigquery",
        "google-cloud-bigquery-storage",
        "pandas-gbq",
        "pytest",
        "tqdm",
        "joblib"
    ],
)
