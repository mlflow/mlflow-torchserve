from argparse import ArgumentParser

from mlflow.deployments import get_deploy_client


def create_deployment(parser_args):
    plugin = get_deploy_client(parser_args["target"])
    config = {
        "MODEL_FILE": parser_args["model_file"],
        "HANDLER": parser_args["handler"],
        "EXTRA_FILES": parser_args["extra_files"],
    }

    if parser_args["export_path"] != "":
        config["EXPORT_PATH"] = parser_args["export_path"]

    result = plugin.create_deployment(
        name=parser_args["deployment_name"],
        model_uri=parser_args["serialized_file_path"],
        config=config,
    )

    print("Deployment {result} created successfully".format(result=result["name"]))


if __name__ == "__main__":
    parser = ArgumentParser(description="Iris Classification example")

    parser.add_argument(
        "--target",
        type=str,
        default="torchserve",
        help="MLflow target (default: torchserve)",
    )

    parser.add_argument(
        "--deployment_name",
        type=str,
        default="iris_classification",
        help="Deployment name (default: iris_classification)",
    )

    parser.add_argument(
        "--model_file",
        type=str,
        default="iris_classification.py",
        help="Model file path (default: iris_classification.py)",
    )

    parser.add_argument(
        "--handler",
        type=str,
        default="iris_handler.py",
        help="Handler file path (default: iris_handler.py)",
    )

    parser.add_argument(
        "--extra_files",
        type=str,
        default="index_to_name.json,model/MLmodel",
        help="List of extra files",
    )

    parser.add_argument(
        "--serialized_file_path",
        type=str,
        default="model",
        help="Pytorch model path (default: model)",
    )

    parser.add_argument(
        "--export_path",
        type=str,
        default="",
        help="Path to model store (default: '')",
    )

    args = parser.parse_args()

    create_deployment(vars(args))
