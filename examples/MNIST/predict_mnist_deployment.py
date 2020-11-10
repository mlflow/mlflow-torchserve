import os
import sys
import matplotlib.pyplot as plt
from torchvision import transforms

from mlflow.deployments import get_deploy_client

if len(sys.argv) != 4:
    raise Exception(
        "\n Format: python predict_mnist_deployment.py <DEPLOYMENT_NAME> <MODEL_URI> <TEST_INPUT_PATH> "
        "\n Ex: python predict_mnist_deployment.py 'mnist_test' 'model' 'test_data/one.png'"
    )

deployment_name = sys.argv[1]
model_file_path = sys.argv[2]
image_path = sys.argv[3]

img = plt.imread(os.path.join(image_path))
mnist_transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

image_tensor = mnist_transforms(img)

plugin = get_deploy_client("torchserve")
prediction = plugin.predict(deployment_name, image_tensor)

print("Prediction Result {}".format(prediction))
