# IMPORTS SECTION #

import logging
import os
import numpy as np
import torch
from torch.autograd import Variable
import json

logger = logging.getLogger(__name__)


# CLASS DEFINITION #


class LinearRegressionHandler(object):
    def __init__(self):
        self.model = None
        self.mapping = None
        self.device = None
        self.initialized = False

    def initialize(self, ctx):
        """
        Loading the saved model from the serialized file
        """

        properties = ctx.system_properties
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu"
        )
        model_dir = properties.get("model_dir")

        # Read model serialize/pt file
        model_pt_path = os.path.join(model_dir, "linear_state_dict.pt")
        # Read model definition file
        model_def_path = os.path.join(model_dir, "linear_model.py")

        if not os.path.isfile(model_def_path):
            model_pt_path = os.path.join(model_dir, "linear_model.pt")
            self.model = torch.load(model_pt_path, map_location=self.device)
        else:
            from linear_model import LinearRegression

            state_dict = torch.load(model_pt_path, map_location=self.device)
            self.model = LinearRegression(1, 1)
            self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

        logger.debug("Model file %s loaded successfully", model_pt_path)
        self.initialized = True

    def preprocess(self, data):
        """
        Preprocess the input to tensor and reshape it to be used as input to the network
        """
        data = data[0]
        image = data.get("data")
        if image is None:
            image = data.get("body")
            image = image.decode("utf-8")
            number = float(json.loads(image)["data"][0])
        else:
            number = float(image)

        np_data = np.array(number, dtype=np.float32)
        np_data = np_data.reshape(-1, 1)
        data_tensor = torch.from_numpy(np_data)
        return data_tensor

    def inference(self, num):

        """
        Does inference / prediction on the preprocessed input and returns the output
        """

        self.model.eval()
        inputs = Variable(num).to(self.device)
        outputs = self.model.forward(inputs)
        return [outputs.detach().item()]

    def postprocess(self, inference_output):

        """
        Does post processing on the output returned from the inference method
        """
        return inference_output


# CLASS INITIALIZATION #

_service = LinearRegressionHandler()


def handle(data, context):

    """
    Default handler for the inference api which takes two parameters data and context
    and returns the predicted output
    """

    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    data = _service.preprocess(data)
    data = _service.inference(data)
    data = _service.postprocess(data)
    print("Data: {}".format(data))
    return data
