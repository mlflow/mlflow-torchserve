import base64
import json
from argparse import ArgumentParser

from mlflow.deployments import get_deploy_client


def predict(parser_args):
    plugin = get_deploy_client(parser_args["target"])
    image = open(parser_args["input_file_path"], "rb")  # open binary file in read mode
    image_read = image.read()
    image_64_encode = base64.b64encode(image_read)
    bytes_array = image_64_encode.decode("utf-8")
    request = {"data": str(bytes_array)}

    inference_type = parser_args["inference_type"]
    if inference_type == "explanation":
        result = plugin.explain(parser_args["deployment_name"], json.dumps(request))
    else:
        result = plugin.predict(parser_args["deployment_name"], json.dumps(request))

    print("Prediction Result {}".format(result))

    output_path = parser_args["output_file_path"]
    if output_path:
        with open(output_path, "w") as fp:
            fp.write(result)


if __name__ == "__main__":
    parser = ArgumentParser(description="Cifar10 classification example")

    parser.add_argument(
        "--target",
        type=str,
        default="torchserve",
        help="MLflow target (default: torchserve)",
    )

    parser.add_argument(
        "--deployment_name",
        type=str,
        default="cifar_test",
        help="Deployment name (default: cifar_test)",
    )

    parser.add_argument(
        "--input_file_path",
        type=str,
        default="test_data/kitten.png",
        help="Path to input image for prediction (default: test_data/one.png)",
    )

    parser.add_argument(
        "--output_file_path",
        type=str,
        default="",
        help="output path to write the result",
    )

    parser.add_argument(
        "--inference_type",
        type=str,
        default="predict",
        help="Option to run prediction/explanation",
    )

    args = parser.parse_args()

    predict(vars(args))
