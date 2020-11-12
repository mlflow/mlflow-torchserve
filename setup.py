from setuptools import setup, find_packages


setup(
    name="mlflow-torchserve",
    version="0.1.0",
    description="Torch Serve Mlflow Deployment",
    long_description=open("README.md").read(),
    packages=find_packages(),
    # Require MLflow as a dependency of the plugin, so that plugin users can simply install
    # the plugin & then immediately use it with MLflow
    install_requires=[
        "mlflow>=1.12.0",
        "torchserve",
        "torch-model-archiver",
    ],
    entry_points={"mlflow.deployments": "torchserve=mlflow_torchserve"},
)
