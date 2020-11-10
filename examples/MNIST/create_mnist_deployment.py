import sys

from mlflow.deployments import get_deploy_client

if len(sys.argv) != 3:
    raise Exception(
        "Deployment name not found. \n "
        "Format: python create_mnist_deployment.py <DEPLOYMENT_NAME> <MODEL_FILE_PATH> \n "
        "Ex: python create_mnist_deployment.py 'mnist_test' 'model'"
    )

deployment_name = sys.argv[1]
model_file_path = sys.argv[2]


plugin = get_deploy_client("torchserve")
config = {"HANDLER": "mnist_handler.py"}
result = plugin.create_deployment(name=deployment_name, model_uri=model_file_path, config=config)

print("Deployment {result} created successfully".format(result=result["name"]))
