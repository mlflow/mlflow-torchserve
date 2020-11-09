import ast
import logging

import numpy as np
import torch
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class IRISClassifierHandler(BaseHandler):
    """
    IRISClassifier handler class. This handler takes an input tensor and
    output the type of iris based on the input
    """

    def __init__(self):
        super(IRISClassifierHandler, self).__init__()

    def preprocess(self, data):
        """
        preprocessing step - Reads the input array and converts it to tensor

        :param data: Input to be passed through the layers for prediction

        :return: output - Preprocessed input
        """

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

        if self.mapping:
            return [self.mapping[str(inference_output[0])]]
        return inference_output


_service = IRISClassifierHandler()
