"""Cifar10 data module."""
import os
import subprocess
from pathlib import Path

import lightning as L
import torchvision
import webdataset as wds
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms


class CIFAR10DataModule(L.LightningDataModule):  # pylint: disable=too-many-instance-attributes
    """Data module class."""

    def __init__(self, num_samples_train=39, num_samples_val=9, num_samples_test=9):
        """Initialization of inherited lightning data module."""
        super(CIFAR10DataModule, self).__init__()  # pylint: disable=super-with-arguments

        self.num_samples_train = num_samples_train
        self.num_samples_val = num_samples_val
        self.num_samples_test = num_samples_test
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.valid_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                self.normalize,
            ]
        )

        self.train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def prepare_data(self):
        """Implementation of abstract class."""
        output_path = "output/processing"
        Path(output_path).mkdir(parents=True, exist_ok=True)

        trainset = torchvision.datasets.CIFAR10(root="./", train=True, download=True)
        testset = torchvision.datasets.CIFAR10(root="./", train=False, download=True)

        Path(output_path + "/train").mkdir(parents=True, exist_ok=True)
        Path(output_path + "/val").mkdir(parents=True, exist_ok=True)
        Path(output_path + "/test").mkdir(parents=True, exist_ok=True)

        RANDOM_SEED = 25
        y = trainset.targets
        trainset, valset, y_train, y_val = train_test_split(
            trainset, y, stratify=y, shuffle=True, test_size=0.2, random_state=RANDOM_SEED
        )

        for name in [(trainset, "train"), (valset, "val"), (testset, "test")]:
            with wds.ShardWriter(
                output_path + "/" + str(name[1]) + "/" + str(name[1]) + "-%d.tar", maxcount=1000
            ) as sink:
                for index, (image, cls) in enumerate(name[0]):
                    sink.write({"__key__": "%06d" % index, "ppm": image, "cls": cls})

        entry_point = ["ls", "-R", output_path]
        run_code = subprocess.run(
            entry_point, stdout=subprocess.PIPE
        )  # pylint: disable=subprocess-run-check
        print(run_code.stdout)

    @staticmethod
    def get_num_files(input_path):
        """Gets num files.
        Args:
             input_path : path to input
        """
        return len(os.listdir(input_path)) - 1

    def setup(self, stage=None):
        """Downloads the data, parse it and split the data into train, test,
        validation data.
        Args:
            stage: Stage - training or testing
        """

        data_path = "output/processing"

        train_base_url = data_path + "/train"
        val_base_url = data_path + "/val"
        test_base_url = data_path + "/test"

        train_count = self.num_samples_train
        val_count = self.num_samples_val
        test_count = self.num_samples_test

        if not train_count:
            train_count = self.get_num_files(train_base_url)

        if not val_count:
            val_count = self.get_num_files(val_base_url)

        if not test_count:
            test_count = self.get_num_files(test_base_url)

        train_url = "{}/{}-{}".format(train_base_url, "train", "{0.." + str(train_count) + "}.tar")
        valid_url = "{}/{}-{}".format(val_base_url, "val", "{0.." + str(val_count) + "}.tar")
        test_url = "{}/{}-{}".format(test_base_url, "test", "{0.." + str(test_count) + "}.tar")

        self.train_dataset = (
            wds.WebDataset(train_url, handler=wds.warn_and_continue)
            .shuffle(100)
            .decode("pil")
            .rename(image="ppm;jpg;jpeg;png", info="cls")
            .map_dict(image=self.train_transform)
            .to_tuple("image", "info")
            .batched(40)
        )

        self.valid_dataset = (
            wds.WebDataset(valid_url, handler=wds.warn_and_continue)
            .shuffle(100)
            .decode("pil")
            .rename(image="ppm", info="cls")
            .map_dict(image=self.valid_transform)
            .to_tuple("image", "info")
            .batched(20)
        )

        self.test_dataset = (
            wds.WebDataset(test_url, handler=wds.warn_and_continue)
            .shuffle(100)
            .decode("pil")
            .rename(image="ppm", info="cls")
            .map_dict(image=self.valid_transform)
            .to_tuple("image", "info")
            .batched(20)
        )

    def create_data_loader(self, dataset, batch_size, num_workers):  # pylint: disable=no-self-use
        """Creates data loader."""
        return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    def train_dataloader(self):
        """Train Data loader.
        Returns:
             output - Train data loader for the given input
        """
        self.train_data_loader = self.create_data_loader(
            dataset=self.train_dataset,
            batch_size=None,
            num_workers=3,
        )
        return self.train_data_loader

    def val_dataloader(self):
        """Validation Data Loader.
        Returns:
             output - Validation data loader for the given input
        """
        self.val_data_loader = self.create_data_loader(
            dataset=self.valid_dataset,
            batch_size=None,
            num_workers=3,
        )
        return self.val_data_loader

    def test_dataloader(self):
        """Test Data Loader.
        Returns:
             output - Test data loader for the given input
        """
        self.test_data_loader = self.create_data_loader(
            dataset=self.test_dataset,
            batch_size=None,
            num_workers=3,
        )
        return self.test_data_loader
