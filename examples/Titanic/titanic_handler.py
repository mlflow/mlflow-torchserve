from titanic import TitanicSimpleNNModel
import json
import logging
import os
import torch
from ts.torch_handler.base_handler import BaseHandler
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class TitanicHandler(BaseHandler):
    """
    Titanic handler class for titanic classifier .
    """

    def __init__(self):
        super(TitanicHandler, self).__init__()
        self.initialized = False
        self.feature_names = None
        self.inference_output = []
        self.predicted_idx = None
        self.out_probs = None
        self.delta = None
        self.input_file_path = None

    def initialize(self, ctx):
        """In this initialize function, the Titanic trained model is loaded and
        the Integrated Gradients Algorithm for Captum Explanations
        is initialized here.

        Args:
            ctx (context): It is a JSON Object containing information
            pertaining to the model artifacts parameters.
        """
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        print("Model dir is {}".format(model_dir))
        serialized_file = self.manifest["model"]["serializedFile"]
        model_pt_path = os.path.join(model_dir, serialized_file)
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu"
        )

        self.model = TitanicSimpleNNModel()
        self.model.load_state_dict(torch.load(model_pt_path))
        self.model.to(self.device)
        self.model.eval()

        logger.info("Titanic model from path %s loaded successfully", model_dir)

        # Read the mapping file, index to object name
        mapping_file_path = os.path.join(model_dir, "index_to_name.json")
        if os.path.isfile(mapping_file_path):
            print("Mapping file present")
            with open(mapping_file_path) as f:
                self.mapping = json.load(f)
        else:
            print("Mapping file missing")
            logger.warning("Missing the index_to_name.json file.")

        # ------------------------------- Captum initialization ----------------------------#
        self.ig = IntegratedGradients(self.model)
        self.initialized = True

    def preprocess(self, data):
        """Basic text preprocessing, based on the user's chocie of application mode.

        Args:
            data (csv): The Input data in the form of csv is passed on to the preprocess
            function.

        Returns:
            list : The preprocess function returns a list of Tensor and feature names
        """
        self.input_file_path = data[0]["body"]["input_file_path"][0]
        data = pd.read_csv(self.input_file_path)
        self.feature_names = list(data.columns)
        data = data.to_numpy()
        data = torch.from_numpy(data).type(torch.FloatTensor)
        return data

    def inference(self, data):
        """Predict the class (survived or not survived) of the received input json file using the
        serialized model.

        Args:
            input_batch (list): List of Tensors from the pre-process function is passed here

        Returns:
            list : It returns a list of the predicted value for the input test record
        """
        data = data.to(self.device)
        self.out_probs = self.model(data)
        self.predicted_idx = self.out_probs.argmax(1).item()
        prediction = self.mapping[str(self.predicted_idx)]
        self.inference_output.append(prediction)
        logger.info("Model predicted: '%s'", prediction)
        return [prediction]

    def postprocess(self, inference_output):
        """Post Process Function converts the predicted response into Torchserve readable format.

        Args:
            inference_output (list): It contains the predicted response of the input record.
        Returns:
            (list): Returns a list of the Predictions and Explanations.
        """
        return inference_output

    def get_insights(
        self, input_tensor, title="Average Feature Importances", axis_title="Features"
    ):
        """This function calls the integrated gradient to the feature importance

        Args:
            data(tensor):
            target (int): The Target can be set to any acceptable label under the user's discretion.

        Returns:
            (list): Returns a dict of feature names and their importances
        """
        ig = IntegratedGradients(self.model)
        input_tensor.requires_grad_()
        input_tensor = input_tensor.to(self.device)
        attr, self.delta = ig.attribute(input_tensor, target=1, return_convergence_delta=True)
        attr = attr.cpu().detach().numpy()
        importances = np.mean(attr, axis=0)
        feature_imp_dict = {}
        for i in range(len(self.feature_names)):
            feature_imp_dict[str(self.feature_names[i])] = importances[i]
        x_pos = np.arange(len(self.feature_names))
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x_pos, importances, align="center")
        ax.set(title=title, xlabel=axis_title)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(self.feature_names, rotation="vertical")
        path = os.path.join(
            os.path.dirname(os.path.abspath(self.input_file_path)), "attributions_imp.png"
        )
        print("path of the saved image", path)

        plt.savefig(path)
        logger.info("Saved attributions image")
        return [feature_imp_dict]

    def explain_handle(self, data_preprocess, raw_data):
        """Captum explanations handler
        Args:
            data_preprocess (Torch Tensor): Preprocessed data to be used for captum
            raw_data (list): The unprocessed data to get target from the request
        Returns:
            dict : A dictionary response with the explanations response.
        """
        output_explain = self.get_insights(data_preprocess)
        return output_explain
