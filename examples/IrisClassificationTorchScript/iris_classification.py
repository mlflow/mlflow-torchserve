import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import seed_everything
from lightning.pytorch.cli import LightningCLI
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader


class IrisClassification(L.LightningModule):
    def __init__(self):
        super(IrisClassification, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 3)
        self.val_outputs = []
        self.test_outputs = []

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(x, dim=0)
        return x

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
        self.val_outputs.append(loss)
        return {"val_step_loss": loss}

    def on_validation_epoch_end(self):
        """
        Computes average validation loss
        """
        avg_loss = torch.stack(self.val_outputs).mean()
        self.log("val_loss", avg_loss, sync_dist=True)
        self.val_outputs.clear()

    def test_step(self, test_batch, batch_idx):
        """
        Performs test and computes test accuracy
        """

        x, y = test_batch
        output = self.forward(x)
        a, y_hat = torch.max(output, dim=1)
        test_acc = accuracy_score(y_hat.cpu(), y.cpu())
        self.test_outputs.append(torch.tensor(test_acc))
        return {"test_acc": torch.tensor(test_acc)}

    def on_test_epoch_end(self):
        """
        Computes average test accuracy score
        """
        avg_test_acc = torch.stack(self.test_outputs).mean()
        self.log("avg_test_acc", avg_test_acc, sync_dist=True)
        self.test_outputs.clear()

    def configure_optimizers(self):
        """
        Creates and returns Optimizer
        """

        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.05, momentum=0.9)
        return self.optimizer


class IrisDataModule(L.LightningDataModule):
    def __init__(self):
        super().__init__()

    def prepare_data(self):
        """
        Implementation of abstract class
        """

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
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

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=4)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=4)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=4)


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
        scripted_model = torch.jit.script(cli.trainer.model)
        torch.jit.save(scripted_model, "iris_ts.pt")


if __name__ == "__main__":
    cli_main()
