import os
import sys
import matplotlib.pyplot as plt
from torchvision import transforms

from mlflow.deployments import get_deploy_client

if len(sys.argv) != 2:
    raise Exception("Model file path not found. \n Ex: python predict.py <MODEL_URI>")

model_file_path = sys.argv[1]
img = plt.imread(os.path.join(os.getcwd(), "test_data/one.png"))
mnist_transforms = transforms.Compose([transforms.ToTensor()])

image = mnist_transforms(img)

plugin = get_deploy_client("torchserve")
config = {"HANDLER": "mnist_handler.py"}
plugin.create_deployment(name="mnist_test", model_uri=model_file_path, config=config)
prediction = plugin.predict("mnist_test", image)

print("Prediction Result {}".format(prediction))
