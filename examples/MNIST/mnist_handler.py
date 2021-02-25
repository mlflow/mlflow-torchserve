import json
import logging
import os

import torch
from ts.torch_handler.image_classifier import ImageClassifier

logger = logging.getLogger(__name__)


class MNISTDigitHandler(ImageClassifier):
    """
    MNISTDigitClassifier handler class. This handler takes a greyscale image
    and returns the digit in that image.
    """

    def __init__(self):
        super(MNISTDigitHandler, self).__init__()
        self.mapping_file_path = None

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
        model_pt_path = os.path.join(model_dir, "state_dict.pth")
        from mnist_model import LightningMNISTClassifier

        state_dict = torch.load(model_pt_path, map_location=self.device)
        self.model = LightningMNISTClassifier()
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        logger.debug("Model file %s loaded successfully", model_pt_path)

        self.mapping_file_path = os.path.join(model_dir, "index_to_name.json")

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

        image = image.decode("utf-8")
        image = torch.Tensor(json.loads(image)["data"])
        return image

    def inference(self, img):
        """
        Predict the class (or classes) of an image using a trained deep learning model
        :param img: Input to be passed through the layers for prediction
        :return: output - Predicted label for the given input
        """
        # Convert 2D image to 1D vector
        # img = np.expand_dims(img, 0)
        # img = torch.from_numpy(img)
        self.model.eval()
        inputs = img.to(self.device)
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

        if self.mapping_file_path:
            with open(self.mapping_file_path) as json_file:
                data = json.load(json_file)
            inference_output = [json.dumps(data[inference_output[0]])]
            return inference_output
        return [inference_output]
