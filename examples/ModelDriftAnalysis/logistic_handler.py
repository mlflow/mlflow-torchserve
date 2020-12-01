import logging
import os
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from classifier import Classification
from torch.nn import functional as F

logger = logging.getLogger(__name__)


### CLASS DEFINITION ###


class LogisticHandler(object):
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
        model_pt_path = os.path.join(model_dir, "model.pth")
        # Read model definition file
        model_def_path = os.path.join(model_dir, "classifier.py")
        if not os.path.isfile(model_def_path):
            raise RuntimeError("Missing the model definition file")

        state_dict = torch.load(model_pt_path, map_location=self.device)
        self.model = Classification()
        self.model.state_dict()
        self.model.to(self.device)
        self.model.eval()

        logger.debug("Model file {0} loaded successfully".format(model_pt_path))
        self.initialized = True

    def preprocess_one_row(self, data):
        """
        Process one single image.
        """
        # get the data
        # print("\n\nHANDLER FILE")
        # print(data)
        # print("\n\n")
        # data = data
        print("INPUT DATA TO HANDLER", data)
        data = [float(i) for i in list(data.values())]
        print("data", data)
        data_tensor = torch.Tensor(data).float()
        print("data_tensor", data_tensor)
        data_tensor = data_tensor.unsqueeze(0)
        print("unsqueeze", data_tensor)
        return data_tensor

    def preprocess(self, requests):
        """
        Process all the data from the requests and batch them in a Tensor.
        """
        # print("*" * 100)
        # print(requests)

        data_tensor = [self.preprocess_one_row(req) for req in requests]
        data_tensors = torch.cat(data_tensor)
        return data_tensors

    def inference(self, num):

        """
        Does inference / prediction on the preprocessed input and returns the output
        """
        print("input data to inference", num)
        self.model.eval()
        inputs = Variable(num).to(self.device)

        print("the inputs to the model", inputs)
        outputs = self.model.forward(inputs)
        probs = F.log_softmax(outputs, dim=1)
        print("outputs:", outputs)
        print("probs:", probs)
        preds = torch.argmax(outputs, dim=1)
        print("prediction", preds)
        return [preds.item()]

    def postprocess(self, inference_output):

        """
        Does post processing on the output returned from the inference method
        """
        return inference_output
        # result = {}
        # result["result"] = inference_output
        # return result


### CLASS INITIALIZATION ###

_service = LogisticHandler()


def handle(data, context):

    """
    Default handler for the inference api which takes two parameters data and context
    and returns the predicted output
    """
    # print("starting:", data)
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None
    # print("preprocess:", data)
    data = _service.preprocess(data)
    # print("inference:", data)
    data = _service.inference(data)
    # print("postprocess:", data)
    data = _service.postprocess(data)
    # print("Data: {}".format(data))
    return data
