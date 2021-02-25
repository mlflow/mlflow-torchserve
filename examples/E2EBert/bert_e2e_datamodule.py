# pylint: disable=arguments-differ
# pylint: disable=unused-argument
# pylint: disable=abstract-method

import os
from argparse import ArgumentParser

import pandas as pd
import pytorch_lightning as pl
import requests
import torch
from pytorch_lightning import seed_everything
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchtext.datasets.text_classification import URLS
from torchtext.utils import download_from_url, extract_archive
from transformers import BertTokenizer


class AGNewsDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_length):
        """
        Performs initialization of tokenizer

        :param reviews: AG news text
        :param targets: labels
        :param tokenizer: bert tokenizer
        :param max_length: maximum length of the news text

        """
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """
        :return: returns the number of datapoints in the dataframe

        """
        return len(self.reviews)

    def __getitem__(self, item):
        """
        Returns the review text and the targets of the specified item

        :param item: Index of sample review

        :return: Returns the dictionary of review text, input ids, attention mask, targets
        """
        review = str(self.reviews[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
        )

        return {
            "review_text": review,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "targets": torch.tensor(target, dtype=torch.long),
        }


class BertDataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        """
        Initialization of inherited lightning data module
        """
        super(BertDataModule, self).__init__()
        self.PRE_TRAINED_MODEL_NAME = "bert-base-uncased"
        self.df_train = None
        self.df_val = None
        self.df_test = None
        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None
        self.MAX_LEN = 100
        self.encoding = None
        self.tokenizer = None
        self.args = kwargs
        self.NUM_SAMPLES_COUNT = self.args["num_samples"]
        self.VOCAB_FILE_URL = self.args["vocab_file"]
        self.VOCAB_FILE = "bert_base_uncased_vocab.txt"

    @staticmethod
    def process_label(rating):
        rating = int(rating)
        return rating - 1

    def prepare_data(self):
        """
        Implementation of abstract class
        """

    def setup(self, stage=None):
        """
        Downloads the data, parse it and split the data into train, test, validation data

        :param stage: Stage - training or testing
        """
        # reading  the input
        dataset_tar = download_from_url(URLS["AG_NEWS"], root=".data")
        extracted_files = extract_archive(dataset_tar)

        for fname in extracted_files:
            if fname.endswith("train.csv"):
                train_csv_path = fname

        df = pd.read_csv(train_csv_path)

        df.columns = ["label", "title", "description"]
        df.sample(frac=1)
        df = df.iloc[: self.NUM_SAMPLES_COUNT]

        df["label"] = df.label.apply(self.process_label)

        if not os.path.isfile(self.VOCAB_FILE):
            filePointer = requests.get(self.VOCAB_FILE_URL, allow_redirects=True)
            if filePointer.ok:
                with open(self.VOCAB_FILE, "wb") as f:
                    f.write(filePointer.content)
            else:
                raise RuntimeError("Error in fetching the vocab file")

        self.tokenizer = BertTokenizer(self.VOCAB_FILE)

        RANDOM_SEED = 42
        seed_everything(RANDOM_SEED)

        df_train, df_test = train_test_split(
            df, test_size=0.2, random_state=RANDOM_SEED, stratify=df["label"]
        )
        df_train, df_val = train_test_split(
            df_train, test_size=0.25, random_state=RANDOM_SEED, stratify=df_train["label"]
        )

        self.df_train = df_train
        self.df_test = df_test
        self.df_val = df_val

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Returns the review text and the targets of the specified item

        :param parent_parser: Application specific parser

        :return: Returns the augmented arugument parser
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--batch-size",
            type=int,
            default=16,
            metavar="N",
            help="input batch size for training (default: 16)",
        )
        parser.add_argument(
            "--num-workers",
            type=int,
            default=3,
            metavar="N",
            help="number of workers (default: 3)",
        )
        return parser

    def create_data_loader(self, df, tokenizer, max_len, batch_size):
        """
        Generic data loader function

        :param df: Input dataframe
        :param tokenizer: bert tokenizer
        :param max_len: Max length of the news datapoint
        :param batch_size: Batch size for training

        :return: Returns the constructed dataloader
        """
        ds = AGNewsDataset(
            reviews=df.description.to_numpy(),
            targets=df.label.to_numpy(),
            tokenizer=tokenizer,
            max_length=max_len,
        )

        return DataLoader(
            ds, batch_size=self.args["batch_size"], num_workers=self.args["num_workers"]
        )

    def train_dataloader(self):
        """
        :return: output - Train data loader for the given input
        """
        self.train_data_loader = self.create_data_loader(
            self.df_train, self.tokenizer, self.MAX_LEN, self.args["batch_size"]
        )
        return self.train_data_loader

    def val_dataloader(self):
        """
        :return: output - Validation data loader for the given input
        """
        self.val_data_loader = self.create_data_loader(
            self.df_val, self.tokenizer, self.MAX_LEN, self.args["batch_size"]
        )
        return self.val_data_loader

    def test_dataloader(self):
        """
        :return: output - Test data loader for the given input
        """
        self.test_data_loader = self.create_data_loader(
            self.df_test, self.tokenizer, self.MAX_LEN, self.args["batch_size"]
        )
        return self.test_data_loader


#
if __name__ == "__main__":
    pass
