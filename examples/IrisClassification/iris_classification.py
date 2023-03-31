# pylint: disable=W0221
# pylint: disable=W0613
# pylint: disable=W0223
import os
import shutil
from argparse import ArgumentParser

import lightning as L
import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import seed_everything
from lightning.pytorch.cli import LightningCLI
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec
from sklearn.datasets import load_iris
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchmetrics import Accuracy


class IrisClassification(L.LightningModule):
    def __init__(self, learning_rate=0.01):
        super(IrisClassification, self).__init__()

        self.train_acc = Accuracy(task="multiclass", num_classes=3)
        self.val_acc = Accuracy(task="multiclass", num_classes=3)
        self.test_acc = Accuracy(task="multiclass", num_classes=3)
        self.learning_rate = learning_rate

        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 3)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.learning_rate)

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


class IrisDataModule(L.LightningDataModule):
    def __init__(self, batch_size=64, num_workers=3):
        """
        Initialization of inherited lightning data module
        """
        super(IrisDataModule, self).__init__()

        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.batch_size = batch_size
        self.num_workers = num_workers

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

        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def train_dataloader(self):
        train_loader = self.create_data_loader(dataset=self.train_set)
        return train_loader

    def val_dataloader(self):
        validation_loader = self.create_data_loader(dataset=self.val_set)
        return validation_loader

    def test_dataloader(self):
        test_loader = self.create_data_loader(dataset=self.test_set)
        return test_loader


def cli_main():
    cli = LightningCLI(
        IrisClassification,
        IrisDataModule,
        run=False,
        save_config_callback=None,
    )
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)
    if cli.trainer.global_rank == 0:
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

        if os.path.exists("model"):
            shutil.rmtree("model")
        mlflow.pytorch.save_model(cli.trainer.lightning_module, "model", signature=signature)


if __name__ == "__main__":
    cli_main()
