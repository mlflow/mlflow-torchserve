#
# Trains an MNIST digit recognizer using PyTorch Lightning
# NOTE: This example requires you to first install
# pytorch-lightning (using pip install pytorch-lightning)
#       and mlflow (using pip install mlflow).
#
# pylint: disable=arguments-differ
# pylint: disable=unused-argument
# pylint: disable=abstract-method

from argparse import ArgumentParser

import mlflow.pytorch
import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.metrics import Accuracy
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        """
        Initialization of inherited lightning data module
        """
        super(MNISTDataModule, self).__init__()
        self.df_train = None
        self.df_val = None
        self.df_test = None
        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None
        self.args = kwargs

        # transforms for images
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    def setup(self, stage=None):
        """
        Downloads the data, parse it and split the data into train, test, validation data

        :param stage: Stage - training or testing
        """

        RANDOM_SEED = 42
        seed_everything(RANDOM_SEED)

        self.df_train = datasets.MNIST(
            "dataset", download=True, train=True, transform=self.transform
        )
        self.df_train, self.df_val = random_split(self.df_train, [55000, 5000])
        self.df_test = datasets.MNIST(
            "dataset", download=True, train=False, transform=self.transform
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--batch-size",
            type=int,
            default=64,
            metavar="N",
            help="input batch size for training (default: 64)",
        )
        parser.add_argument(
            "--num-workers",
            type=int,
            default=3,
            metavar="N",
            help="number of workers (default: 3)",
        )
        return parser

    def create_data_loader(self, df):
        """
        Generic data loader function

        :param df: Input tensor

        :return: Returns the constructed dataloader
        """
        return DataLoader(
            df,
            batch_size=self.args["batch_size"],
            num_workers=self.args["num_workers"],
        )

    def train_dataloader(self):
        """
        :return: output - Train data loader for the given input
        """
        return self.create_data_loader(self.df_train)

    def val_dataloader(self):
        """
        :return: output - Validation data loader for the given input
        """
        return self.create_data_loader(self.df_val)

    def test_dataloader(self):
        """
        :return: output - Test data loader for the given input
        """
        return self.create_data_loader(self.df_test)


class LightningMNISTClassifier(pl.LightningModule):
    def __init__(self, **kwargs):
        """mlflow.start_run()
        Initializes the network
        """
        super(LightningMNISTClassifier, self).__init__()

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 256)
        self.layer_3 = torch.nn.Linear(256, 10)
        self.args = kwargs

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--lr",
            type=float,
            default=0.001,
            metavar="LR",
            help="learning rate (default: 0.001)",
        )
        return parser

    def forward(self, x):
        """
        :param x: Input data

        :return: output - mnist digit label for the input image
        """
        batch_size = x.size()[0]

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)

        # layer 1 (b, 1*28*28) -> (b, 128)
        x = self.layer_1(x)
        x = torch.relu(x)

        # layer 2 (b, 128) -> (b, 256)
        x = self.layer_2(x)
        x = torch.relu(x)

        # layer 3 (b, 256) -> (b, 10)
        x = self.layer_3(x)

        # probability distribution over labels
        x = torch.log_softmax(x, dim=1)

        return x

    def cross_entropy_loss(self, logits, labels):
        """
        Initializes the loss function

        :return: output - Initialized cross entropy loss function
        """
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        """
        Training the data as batches and returns training loss on each batch
        :param train_batch: Batch data
        :param batch_idx: Batch indices

        :return: output - Training loss
        """
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.train_acc(logits, y)
        self.log("train_acc", self.train_acc.compute())
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, val_batch, batch_idx):
        """
        Performs validation of data in batches

        :param val_batch: Batch data
        :param batch_idx: Batch indices

        :return: output - valid step loss
        """
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.val_acc(logits, y)
        self.log("val_acc", self.val_acc.compute())
        self.log("val_loss", loss, sync_dist=True)

    def test_step(self, test_batch, batch_idx):
        """
        Performs test and computes the accuracy of the model

        :param test_batch: Batch data
        :param batch_idx: Batch indices

        :return: output - Testing accuracy
        """
        x, y = test_batch
        output = self.forward(x)
        _, y_hat = torch.max(output, dim=1)

        self.test_acc(y_hat, y)
        self.log("test_acc", self.test_acc.compute())

    def prepare_data(self):
        """
        Prepares the data for training and prediction
        """
        return {}

    def configure_optimizers(self):
        """
        Initializes the optimizer and learning rate scheduler

        :return: output - Initialized optimizer and scheduler
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args["lr"])
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.2,
                patience=2,
                min_lr=1e-6,
                verbose=True,
            ),
            "monitor": "val_loss",
        }
        return [optimizer], [scheduler]


if __name__ == "__main__":
    parser = ArgumentParser(description="PyTorch Autolog Mnist Example")

    # Add trainer specific arguments
    parser.add_argument(
        "--model-save-path", type=str, default="model", help="Path to save mlflow model"
    )

    parser = pl.Trainer.add_argparse_args(parent_parser=parser)
    parser = LightningMNISTClassifier.add_model_specific_args(parent_parser=parser)
    parser = MNISTDataModule.add_model_specific_args(parent_parser=parser)

    args = parser.parse_args()
    dict_args = vars(args)

    if "accelerator" in dict_args:
        if dict_args["accelerator"] == "None":
            dict_args["accelerator"] = None

    mlflow.pytorch.autolog()

    model = LightningMNISTClassifier(**dict_args)

    dm = MNISTDataModule(**dict_args)
    dm.prepare_data()
    dm.setup(stage="fit")

    trainer = pl.Trainer.from_argparse_args(args)

    trainer.fit(model, dm)
    trainer.test()

    trainer.save_checkpoint("model.pth")
