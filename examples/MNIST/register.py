from argparse import ArgumentParser

from mlflow.deployments import get_deploy_client


def register(parser_args):
    plugin = get_deploy_client(parser_args["target"])
    plugin.register_model(mar_file_path="mnist_test.mar")
    print("Registered Successfully")


if __name__ == "__main__":
    parser = ArgumentParser(description="MNIST hand written digits classification example")

    parser.add_argument(
        "--target",
        type=str,
        default="torchserve",
        help="MLflow target (default: torchserve)",
    )

    parser.add_argument(
        "--deployment_name",
        type=str,
        default="mnist_classification",
        help="Deployment name (default: mnist_classification)",
    )

    args = parser.parse_args()

    register(vars(args))
