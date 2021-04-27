# pylint: disable=W0221
# pylint: disable=W0613
# pylint: disable=W0223
import argparse
import os
import shutil
from argparse import ArgumentParser

import mlflow
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec
from pytorch_lightning import seed_everything
from pytorch_lightning.metrics import Accuracy
from sklearn.datasets import load_iris
from torch.utils.data import DataLoader, random_split, TensorDataset


class IrisClassification(pl.LightningModule):
    def __init__(self, **kwargs):
        super(IrisClassification, self).__init__()

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
        self.args = kwargs

        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 3)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Add model specific arguments like learning rate

        :param parent_parser: Application specific parser

        :return: Returns the augmented arugument parser
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--lr",
            type=float,
            default=0.01,
            metavar="LR",
            help="learning rate (default: 0.001)",
        )
        return parser

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.args["lr"])

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        _, y_hat = torch.max(logits, dim=1)
        loss = self.cross_entropy_loss(logits, y)
        self.train_acc(y_hat, y)
        self.log(
            "train_acc",
            self.train_acc.compute(),
            on_step=False,
            on_epoch=True,
        )
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        _, y_hat = torch.max(logits, dim=1)
        loss = F.cross_entropy(logits, y)
        self.val_acc(y_hat, y)
        self.log("val_acc", self.val_acc.compute())
        self.log("val_loss", loss, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        _, y_hat = torch.max(logits, dim=1)
        self.test_acc(y_hat, y)
        self.log("test_acc", self.test_acc.compute())


class IrisDataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        """
        Initialization of inherited lightning data module
        """
        super(IrisDataModule, self).__init__()

        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.args = kwargs

    def prepare_data(self):
        """
        Implementation of abstract class
        """

    def setup(self, stage=None):
        """
        Downloads the data, parse it and split the data into train, test, validation data

        :param stage: Stage - training or testing
        """
        iris = load_iris()
        df = iris.data
        target = iris["target"]

        data = torch.Tensor(df).float()
        labels = torch.Tensor(target).long()
        RANDOM_SEED = 42
        seed_everything(RANDOM_SEED)

        data_set = TensorDataset(data, labels)
        self.train_set, self.val_set = random_split(data_set, [130, 20])
        self.train_set, self.test_set = random_split(self.train_set, [110, 20])

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Adds model specific arguments batch size and num workers

        :param parent_parser: Application specific parser

        :return: Returns the augmented arugument parser
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--batch-size",
            type=int,
            default=128,
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

    def create_data_loader(self, dataset):
        """
        Generic data loader function

        :param data_set: Input data set

        :return: Returns the constructed dataloader
        """

        return DataLoader(
            dataset, batch_size=self.args["batch_size"], num_workers=self.args["num_workers"]
        )

    def train_dataloader(self):
        train_loader = self.create_data_loader(dataset=self.train_set)
        return train_loader

    def val_dataloader(self):
        validation_loader = self.create_data_loader(dataset=self.val_set)
        return validation_loader

    def test_dataloader(self):
        test_loader = self.create_data_loader(dataset=self.test_set)
        return test_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iris Classification model")

    parser.add_argument(
        "--max_epochs", type=int, default=100, help="number of epochs to run (default: 100)"
    )
    parser.add_argument(
        "--gpus", type=int, default=0, help="Number of gpus - by default runs on CPU"
    )
    parser.add_argument(
        "--save-model",
        type=bool,
        default=True,
        help="For Saving the current Model",
    )
    parser.add_argument(
        "--accelerator",
        type=lambda x: None if x == "None" else x,
        default=None,
        help="Accelerator - (default: None)",
    )

    parser = IrisClassification.add_model_specific_args(parent_parser=parser)
    parser = IrisDataModule.add_model_specific_args(parent_parser=parser)

    args = parser.parse_args()
    dict_args = vars(args)

    dm = IrisDataModule(**dict_args)
    dm.prepare_data()
    dm.setup(stage="fit")

    model = IrisClassification(**dict_args)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, dm)
    trainer.test()

    input_schema = Schema(
        [
            ColSpec("double", "sepal length (cm)"),
            ColSpec("double", "sepal width (cm)"),
            ColSpec("double", "petal length (cm)"),
            ColSpec("double", "petal width (cm)"),
        ]
    )
    output_schema = Schema([ColSpec("long")])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)
    with mlflow.start_run():
        if os.path.exists("model"):
            shutil.rmtree("model")
        mlflow.pytorch.save_model(trainer.get_model(), "model", signature=signature)
