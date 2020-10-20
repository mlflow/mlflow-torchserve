import os

import matplotlib.pyplot as plt
from torchvision import transforms

from mlflow.deployments import get_deploy_client

img = plt.imread(os.path.join(os.getcwd(), "test_data/one.png"))
mnist_transforms = transforms.Compose([
    transforms.ToTensor()
])

image = mnist_transforms(img)

plugin = get_deploy_client("torchserve")
config = {
    'MODEL_FILE': "mnist_model.py",
    'HANDLER_FILE': 'mnist_handler.py'
}
plugin.create_deployment(name="mnist_test", model_uri="mnist_cnn.pt", config=config)
prediction = plugin.predict("mnist_test", image)

print("Prediction Result {}".format(prediction))
