import ast
import logging
import os

import numpy as np
import torch

from iris_classification import IrisClassification

logger = logging.getLogger(__name__)


class IRISClassifierHandler(object):
    """
    IRISClassifier handler class. This handler takes an input tensor and
    output the type of iris based on the input
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
            "cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu"
        )
        model_dir = properties.get("model_dir")

        # Read model serialize/pt file
        model_pt_path = os.path.join(model_dir, "iris.pt")
        # Read model definition file
        model_def_path = os.path.join(model_dir, "iris_classification.py")
        if not os.path.isfile(model_def_path):
            raise RuntimeError("Missing the model definition file")

        state_dict = torch.load(model_pt_path, map_location=self.device)
        self.model = IrisClassification()
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        logger.debug("Model file %s loaded successfully", model_pt_path)
        self.initialized = True

    def preprocess(self, data):
        """
        preprocessing step - Reads the input array and converts it to tensor

        :param data: Input to be passed through the layers for prediction

        :return: output - Preprocessed input
        """

        print(data)
        input_data_str = data[0].get("data")
        if input_data_str is None:
            input_data_str = data[0].get("body")

        input_data = input_data_str.decode("utf-8")
        input_tensor = torch.Tensor(ast.literal_eval(input_data))
        return input_tensor

    def inference(self, input_data):
        """
        Predict the class (or classes) for the given input using a trained deep learning model

        :param input_data: Input to be passed through the layers for prediction

        :return: output - Predicted label for the given input
        """

        self.model.eval()
        inputs = input_data.to(self.device)
        outputs = self.model(inputs)

        predicted_idx = str(np.argmax(outputs.cpu().detach().numpy()))

        return [predicted_idx]

    def postprocess(self, inference_output):
        """
        Does postprocess after inference to be returned to user

        :param inference_output: Output of inference

        :return: output - Output after post processing
        """
        return inference_output


_service = IRISClassifierHandler()


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
