import json
import logging
import os

import numpy as np
import torch
from transformers import BertTokenizer
from ts.torch_handler.base_handler import BaseHandler

from news_classifier import BertNewsClassifier

logger = logging.getLogger(__name__)


class NewsClassifierHandler(BaseHandler):
    """
    NewsClassifierHandler class. This handler takes a review / sentence
    and returns the label as either world / sports / business /sci-tech
    """

    def __init__(self):
        self.model = None
        self.mapping = None
        self.device = None
        self.initialized = False
        self.class_mapping_file = None
        self.VOCAB_FILE = None

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
        # Read model definition file
        model_def_path = os.path.join(model_dir, "news_classifier.py")
        if not os.path.isfile(model_def_path):
            raise RuntimeError("Missing the model definition file")

        self.VOCAB_FILE = os.path.join(model_dir, "bert_base_uncased_vocab.txt")
        if not os.path.isfile(self.VOCAB_FILE):
            raise RuntimeError("Missing the vocab file")

        self.class_mapping_file = os.path.join(model_dir, "class_mapping.json")

        state_dict = torch.load(model_pt_path, map_location=self.device)
        self.model = BertNewsClassifier()
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        logger.debug("Model file %s loaded successfully", model_pt_path)
        self.initialized = True

    def preprocess(self, data):
        """
        Receives text in form of json and converts it into an encoding for the inference stage

        :param data: Input to be passed through the layers for prediction

        :return: output - preprocessed encoding
        """

        text = data[0].get("data")
        if text is None:
            text = data[0].get("body")

        text = text.decode("utf-8")

        tokenizer = BertTokenizer(self.VOCAB_FILE)  # .from_pretrained("bert-base-cased")
        encoding = tokenizer.encode_plus(
            text,
            max_length=32,
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",  # Return PyTorch tensors
            truncation=True,
        )

        return encoding

    def inference(self, encoding):
        """
        Predict the class whether it is Positive / Neutral / Negative

        :param encoding: Input encoding to be passed through the layers for prediction

        :return: output - predicted output
        """

        self.model.eval()
        inputs = encoding.to(self.device)
        outputs = self.model.forward(**inputs)

        out = np.argmax(outputs.cpu().detach())
        return [out.item()]

    def postprocess(self, inference_output):
        """
        Does postprocess after inference to be returned to user

        :param inference_output: Output of inference

        :return: output - Output after post processing
        """
        if os.path.exists(self.class_mapping_file):
            with open(self.class_mapping_file) as json_file:
                data = json.load(json_file)
            inference_output = json.dumps(data[str(inference_output[0])])
            return [inference_output]

        return inference_output
