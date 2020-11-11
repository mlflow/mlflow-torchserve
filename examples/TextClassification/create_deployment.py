from argparse import ArgumentParser
from mlflow.deployments import get_deploy_client


def create_deployment(parser_args):
    plugin = get_deploy_client(parser_args["target"])
    config = {
        "MODEL_FILE": parser_args["model_file"],
        "HANDLER": parser_args["handler"],
        "EXTRA_FILES": "source_vocab.pt,index_to_name.json",
    }
    result = plugin.create_deployment(
        name=parser_args["deployment_name"],
        model_uri=parser_args["serialized_file"],
        config=config,
    )

    print("Deployment {result} created successfully".format(result=result["name"]))


if __name__ == "__main__":
    parser = ArgumentParser(description="Text Classifier Example")

    parser.add_argument(
        "--target",
        type=str,
        default="torchserve",
        help="MLflow target (default: torchserve)",
    )

    parser.add_argument(
        "--deployment_name",
        type=str,
        default="text_classification",
        help="Deployment name (default: text_classification)",
    )

    parser.add_argument(
        "--model_file",
        type=str,
        default="model.py",
        help="Model file path (default: model.py)",
    )

    parser.add_argument(
        "--handler",
        type=str,
        default="text_classifier",
        help="Handler file path (default: text_classifier)",
    )

    parser.add_argument(
        "--extra_files",
        type=str,
        default="source_vocab.pt,index_to_name.json",
        help="List of extra files",
    )

    parser.add_argument(
        "--serialized_file",
        type=str,
        default="model.pt",
        help="Pytorch model path (default: model.pt)",
    )

    args = parser.parse_args()

    create_deployment(vars(args))
