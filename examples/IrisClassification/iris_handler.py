import logging

import numpy as np
import torch
import pandas as pd
import os
import json
from ts.torch_handler.base_handler import BaseHandler
from mlflow.models.model import Model

logger = logging.getLogger(__name__)


class IRISClassifierHandler(BaseHandler):
    """
    IRISClassifier handler class. This handler takes an input tensor and
    output the type of iris based on the input
    """

    def __init__(self):
        super(IRISClassifierHandler, self).__init__()
        self.mlmodel = None

    def preprocess(self, data):
        """
        preprocessing step - Reads the input array and converts it to tensor

        :param data: Input to be passed through the layers for prediction

        :return: output - Preprocessed input
        """
        from mlflow_torchserve.SignatureValidator import SignatureValidator

        data = json.loads(data[0]["data"].decode("utf-8"))
        df = pd.DataFrame(data)

        SignatureValidator(model_meta=self.mlmodel)._enforce_schema(
            df, self.mlmodel.get_input_schema()
        )

        input_tensor = torch.Tensor(list(df.iloc[0]))
        return input_tensor

    def extract_signature(self, mlmodel_file):
        self.mlmodel = Model.load(mlmodel_file)
        model_json = json.loads(Model.to_json(self.mlmodel))

        if "signature" not in model_json.keys():
            raise Exception("Model Signature not found")

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
        model_pt_path = os.path.join(model_dir, "model.pth")

        self.model = torch.load(model_pt_path, map_location=self.device)

        logger.debug("Model file %s loaded successfully", model_pt_path)

        mapping_file_path = os.path.join(model_dir, "index_to_name.json")
        if os.path.exists(mapping_file_path):
            with open(mapping_file_path) as fp:
                self.mapping = json.load(fp)
        mlmodel_file = os.path.join(model_dir, "MLmodel")

        self.extract_signature(mlmodel_file=mlmodel_file)

        self.initialized = True

    def postprocess(self, inference_output):
        """
        Does postprocess after inference to be returned to user

        :param inference_output: Output of inference

        :return: output - Output after post processing
        """

        predicted_idx = str(np.argmax(inference_output.cpu().detach().numpy()))

        if self.mapping:
            return [self.mapping[str(predicted_idx)]]
        return [predicted_idx]


_service = IRISClassifierHandler()
