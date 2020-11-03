from mlflow.deployments import BaseDeploymentClient


class TorchServePlugin(BaseDeploymentClient):
    def __init__(self, uri):
        super().__init__(target_uri=uri)

    def create_deployment(self, name, model_uri, flavor=None, config=None):
        return {"name": name, "flavor": flavor}

    def delete_deployment(self, name):
        pass

    def update_deployment(self, name, model_uri=None, flavor=None, config=None):
        pass

    def list_deployments(self):
        pass

    def get_deployment(self, name):
        pass

    def predict(self, deployment_name, df):
        pass


def run_local(name, model_uri, flavor=None, config=None):
    print("torchserve plugin")


def target_help():
    return "torchserve plugin"
