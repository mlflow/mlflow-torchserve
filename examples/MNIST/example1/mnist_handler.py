import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms
from mnist_model import Net

logger = logging.getLogger(__name__)


class MNISTDigitClassifier(object):
    """
    MNISTDigitClassifier handler class. This handler takes a greyscale image
    and returns the digit in that image.
    """

    def __init__(self):
        self.model = None
        self.mapping = None
        self.device = None
        self.initialized = False

    def initialize(self, ctx):
        """
        First try to load torchscript else load eager mode state_dict based model

        :param ctx: System properties
        """

        properties = ctx.system_properties
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available()
            else "cpu"
        )
        model_dir = properties.get("model_dir")

        # Read model serialize/pt file
        model_pt_path = os.path.join(model_dir, "mnist_cnn.pt")
        # Read model definition file
        model_def_path = os.path.join(model_dir, "mnist_model.py")
        if not os.path.isfile(model_def_path):
            raise RuntimeError("Missing the model definition file")

        state_dict = torch.load(model_pt_path, map_location=self.device)
        self.model = Net()
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        logger.debug("Model file %s loaded successfully", model_pt_path)
        self.initialized = True

    def preprocess(self, data):
        """
        Scales, crops, and normalizes a PIL image for a MNIST model,
         returns an Numpy array

        :param data: Input to be passed through the layers for prediction

        :return: output - Preprocessed image
        """
        image = data[0].get("data")
        if image is None:
            image = data[0].get("body")

        mnist_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        image = image.decode("utf-8")
        image = plt.imread(image)
        image = mnist_transform(image)
        return image

    def inference(self, img):
        """
        Predict the class (or classes) of an image using a trained deep learning model

        :param img: Input to be passed through the layers for prediction

        :return: output - Predicted label for the given input
        """
        # Convert 2D image to 1D vector
        img = np.expand_dims(img, 0)
        img = torch.from_numpy(img)

        self.model.eval()
        inputs = Variable(img).to(self.device)
        outputs = self.model.forward(inputs)

        _, y_hat = outputs.max(1)
        predicted_idx = str(y_hat.item())
        return [predicted_idx]

    def postprocess(self, inference_output):
        """
        Does postprocess after inference to be returned to user

        :param inference_output: Output of inference

        :return: output - Output after post processing
        """
        return inference_output


_service = MNISTDigitClassifier()


def handle(data, context):
    """
    Default function that is called when predict is invoked

    :param data: Input to be passed through the layers for prediction
    :param context: dict containing system properties

    :return: output - Output after postprocess
    """

    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    data = _service.preprocess(data)
    data = _service.inference(data)
    data = _service.postprocess(data)

    return data
