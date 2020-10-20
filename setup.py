from setuptools import setup, find_packages


setup(
    name="mlflow-torchserve",
    version="0.0.1",
    description="Torch Serve Mlflow Deployment",
    packages=find_packages(),
    # Require MLflow as a dependency of the plugin, so that plugin users can simply install
    # the plugin & then immediately use it with MLflow
    install_requires=["mlflow"],
    entry_points={"mlflow.deployments": "torchserve=deploy.serve"},
)
