import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
from mlflow.deployments import get_deploy_client
from torchvision import transforms


def predict(parser_args):
    plugin = get_deploy_client(parser_args["target"])
    img = plt.imread(os.path.join(parser_args["input_file_path"]))
    mnist_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    image_tensor = mnist_transforms(img)
    prediction = plugin.predict(parser_args["deployment_name"], image_tensor)
    print("Prediction Result {}".format(prediction.to_json()))


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

    parser.add_argument(
        "--input_file_path",
        type=str,
        default="test_data/one.png",
        help="Path to input image for prediction (default: test_data/one.png)",
    )

    args = parser.parse_args()

    predict(vars(args))
