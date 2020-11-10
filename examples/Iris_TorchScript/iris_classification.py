import argparse

from sklearn.datasets import load_iris
import torch.nn as nn
from torch.utils.data import TensorDataset
import pytorch_lightning as pl
import torch
import mlflow.pytorch
from sklearn.metrics import accuracy_score
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split


class IrisClassification(pl.LightningModule):
    def __init__(self):
        super(IrisClassification, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 3)

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
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        """
        Computes average validation accuracy
        """
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}

    def test_step(self, test_batch, batch_idx):
        """
        Performs test and computes test accuracy
        """

        x, y = test_batch
        output = self.forward(x)
        a, y_hat = torch.max(output, dim=1)
        test_acc = accuracy_score(y_hat.cpu(), y.cpu())
        return {"test_acc": torch.tensor(test_acc)}

    def test_epoch_end(self, outputs):
        """
        Computes average test accuracy score
        """
        avg_test_acc = torch.stack([x["test_acc"] for x in outputs]).mean()
        return {"avg_test_acc": avg_test_acc}

    def configure_optimizers(self):
        """
        Creates and returns Optimizer
        """

        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.05, momentum=0.9)
        return self.optimizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-epochs",
        default=3,
        help="Describes the number of times a neural network has to be trained",
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
    args = parser.parse_args()
    mlflow.pytorch.autolog()
    trainer = pl.Trainer(max_epochs=int(args.max_epochs))
    model = IrisClassification()
    import iris_datamodule

    dm = iris_datamodule.IRISDataModule()
    dm.setup("fit")
    trainer.fit(model, dm)
    testloader = dm.setup("test")
    trainer.test(datamodule=testloader)
    scripted_model = torch.jit.script(model)
    torch.jit.save(scripted_model, "iris_ts.pt")
