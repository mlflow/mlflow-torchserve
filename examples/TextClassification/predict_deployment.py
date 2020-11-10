import json
from argparse import ArgumentParser

from mlflow.deployments import get_deploy_client


def predict_deployment(parser_args):

    with open(parser_args["input_file_path"], "r") as fp:
        text = fp.read()
        plugin = get_deploy_client(parser_args["target"])
        prediction = plugin.predict(parser_args["deployment_name"], json.dumps(text))
        print("Prediction Result {}".format(prediction))


if __name__ == "__main__":
    parser = ArgumentParser(description="PyTorch Autolog Mnist Example")

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
        "--input_file_path",
        type=str,
        default="sample_text.txt",
        help="Path to input text file for prediction (default: sample_text.txt)",
    )

    args = parser.parse_args()

    predict_deployment(vars(args))
