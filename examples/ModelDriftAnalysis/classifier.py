import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data import TensorDataset
import mlflow.pytorch
from pytorch_lightning.metrics.functional import accuracy


class DataModule(pl.LightningDataModule):
    def __init__(self, dataframe):
        super().__init__()
        self.dataframe = dataframe

    def prepare_data(self):
        df1 = self.dataframe
        # df1 = df1.drop(['id','date'], axis=1)
        df1 = df1.set_index("dates")
        data = df1.drop("class", axis=1)

        data_array = data.to_numpy()
        data_tensor = torch.Tensor(data_array).float()

        labels = torch.Tensor(df1["class"].values).long()

        full_dataset = TensorDataset(data_tensor, labels)

        return full_dataset

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.full_data = self.prepare_data()
            self.train_set, self.val_set = self.split_train_dataset(self.full_data)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.train_set, self.test_set = self.split_train_dataset(self.full_data)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=8)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=8)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=8)

    def split_train_dataset(self, dataset, frac_split=0.10):
        train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=frac_split)
        train_dataset = Subset(dataset, train_idx)
        val_or_test_dataset = Subset(dataset, val_idx)
        return train_dataset, val_or_test_dataset


class Classification(pl.LightningModule):
    def __init__(self, input_size=4, num_classes=2, **kwargs):
        super(Classification, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
        self.lr = 0.0
        self.nesterov = False
        self.momentum = 0.0
        self.lr = kwargs.get("kwargs", {}).get("lr")
        self.momentum = kwargs.get("kwargs", {}).get("momentum")
        self.weight_decay = kwargs.get("kwargs", {}).get("weight_decay")
        self.nesterov = kwargs.get("kwargs", {}).get("nesterov")

    def forward(self, x):
        out = self.linear(x)
        return F.log_softmax(out, dim=1)

    def cross_entropy_loss(self, logits, labels):
        """
        Loss Fn to compute loss
        """
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        """
        training the data as batches and returns training loss on each batch
        """
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        return {"loss": loss}

    def validation_step(self, val_batch, batch_idx):
        """
        Performs validation of data in batches
        """
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        """
        Computes average validation accuracy
        """
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("val_loss", avg_loss)

    def test_step(self, test_batch, batch_idx):
        """
        Performs test and computes test accuracy
        """
        x, y = test_batch
        output = self.forward(x)
        loss = F.cross_entropy(output, y)
        a, y_hat = torch.max(output, dim=1)
        test_acc = accuracy(y_hat.cpu(), y.cpu())
        self.log("test_loss", loss)
        self.log("test_acc", test_acc)
        return {"test_loss": loss, "test_acc": test_acc}

    def test_epoch_end(self, outputs):
        """
        Computes average test accuracy score
        """
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        avg_test_acc = torch.stack([x["test_acc"] for x in outputs]).mean()
        self.log("avg_test_loss", avg_loss)
        self.log("avg_test_acc", avg_test_acc)

    def configure_optimizers(self):
        """
        Creates and returns Optimizer
        """

        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            nesterov=self.nesterov,
            weight_decay=self.weight_decay,
        )
        return self.optimizer


def train_evaluate(parameterization=None, dm=None, model=None, max_epochs=1):

    trainer = pl.Trainer(max_epochs=max_epochs)
    mlflow.pytorch.autolog()
    dm.prepare_data()
    dm.setup("fit")
    trainer.fit(model, dm)
    dm.setup("test")
    trainer.test(datamodule=dm)
    testing_metrics = trainer.callback_metrics
    test_accuracy = testing_metrics.get("avg_test_acc")
    return test_accuracy
