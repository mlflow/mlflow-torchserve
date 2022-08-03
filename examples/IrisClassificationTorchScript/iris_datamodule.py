import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import torch
from sklearn.datasets import load_iris
from torch.utils.data import TensorDataset
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader


class IRISDataModule(pl.LightningDataModule):
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
