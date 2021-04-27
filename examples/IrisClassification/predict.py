import os
import json
import torch
import ast
from argparse import ArgumentParser

from mlflow.deployments import get_deploy_client


def convert_input_to_tensor(data):
    data = json.loads(data).get("data")
    input_tensor = torch.Tensor(ast.literal_eval(data[0]))
    return input_tensor


def predict(parser_args):
    plugin = get_deploy_client(parser_args["target"])
    input_file = parser_args["input_file_path"]
    if not os.path.exists(input_file):
        raise Exception("Unable to locate inuput file : {}".format(input_file))
    else:
        with open(input_file) as fp:
            input_data = fp.read()

    data = json.loads(input_data).get("data")
    import pandas as pd

    df = pd.read_json(data[0])
    for column in df.columns:
        df[column].astype("double")

    prediction = plugin.predict(deployment_name=parser_args["deployment_name"], df=input_data)
    print("Prediction Result {}".format(prediction))


if __name__ == "__main__":
    parser = ArgumentParser(description="Iris Classifiation Model")

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
        "--input_file_path",
        type=str,
        default="sample.json",
        help="Path to input image for prediction (default: sample.json)",
    )

    parser.add_argument(
        "--mlflow-model-uri",
        type=str,
        default="model",
        help="MLFlow model URI)",
    )
    args = parser.parse_args()

    predict(vars(args))
