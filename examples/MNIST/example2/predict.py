import os
import sys
import matplotlib.pyplot as plt
from torchvision import transforms

from mlflow.deployments import get_deploy_client

if len(sys.argv) != 3:
    raise Exception("Invalid Input. Ex: python predict.py <MODEL_FILE_PATH> <HANDLER_PATH>")

img = plt.imread(os.path.join(os.getcwd(), "test_data/one.png"))
mnist_transforms = transforms.Compose([transforms.ToTensor()])

image = mnist_transforms(img)

plugin = get_deploy_client("torchserve")
config = {"MODEL_FILE": "mnist_model.py", "HANDLER": "mnist_handler.py"}
plugin.create_deployment(name="mnist_test", model_uri="mnist_cnn.pt", config=config)
prediction = plugin.predict("mnist_test", image)

print("Prediction Result {}".format(prediction))
