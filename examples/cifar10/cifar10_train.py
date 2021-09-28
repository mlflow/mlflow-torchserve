"""Cifar10 training module."""
import mlflow.pytorch
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from argparse import ArgumentParser, Namespace
from pytorch_lightning.metrics import Accuracy
from torch import nn
from torchvision import models
from pytorch_lightning.plugins import DeepSpeedPlugin


class CIFAR10Classifier(
    pl.LightningModule
):  # pylint: disable=too-many-ancestors,too-many-instance-attributes
    """Cifar10 model class."""

    def __init__(self, **kwargs):
        """Initializes the network, optimizer and scheduler."""
        super(CIFAR10Classifier, self).__init__()  # pylint: disable=super-with-arguments
        self.model_conv = models.resnet50(pretrained=True)
        for param in self.model_conv.parameters():
            param.requires_grad = False
        num_ftrs = self.model_conv.fc.in_features
        num_classes = 10
        self.model_conv.fc = nn.Linear(num_ftrs, num_classes)

        self.scheduler = None
        self.optimizer = None
        self.args = kwargs

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        self.preds = []
        self.target = []

    def forward(self, x_var):
        """Forward function."""
        out = self.model_conv(x_var)
        return out

    def training_step(self, train_batch, batch_idx):
        """Training Step
        Args:
             train_batch : training batch
             batch_idx : batch id number
        Returns:
            train accuracy
        """
        if batch_idx == 0:
            self.reference_image = (train_batch[0][0]).unsqueeze(
                0
            )  # pylint: disable=attribute-defined-outside-init
            # self.reference_image.resize((1,1,28,28))
            print("\n\nREFERENCE IMAGE!!!")
            print(self.reference_image.shape)
        x_var, y_var = train_batch
        output = self.forward(x_var)
        _, y_hat = torch.max(output, dim=1)
        loss = F.cross_entropy(output, y_var)
        self.log("train_loss", loss)
        self.train_acc(y_hat, y_var)
        self.log("train_acc", self.train_acc.compute())
        return {"loss": loss}

    def test_step(self, test_batch, batch_idx):
        """Testing step
        Args:
             test_batch : test batch data
             batch_idx : tests batch id
        Returns:
             test accuracy
        """

        x_var, y_var = test_batch
        output = self.forward(x_var)
        _, y_hat = torch.max(output, dim=1)
        loss = F.cross_entropy(output, y_var)
        accelerator = self.args.get("accelerator", None)
        if accelerator is not None:
            self.log("test_loss", loss, sync_dist=True)
        else:
            self.log("test_loss", loss)
        self.test_acc(y_hat, y_var)
        self.preds += y_hat.tolist()
        self.target += y_var.tolist()

        self.log("test_acc", self.test_acc.compute())
        return {"test_acc": self.test_acc.compute()}

    def validation_step(self, val_batch, batch_idx):
        """Testing step.
        Args:
             val_batch : val batch data
             batch_idx : val batch id
        Returns:
             validation accuracy
        """

        x_var, y_var = val_batch
        output = self.forward(x_var)
        _, y_hat = torch.max(output, dim=1)
        loss = F.cross_entropy(output, y_var)
        accelerator = self.args.get("accelerator", None)
        if accelerator is not None:
            self.log("val_loss", loss, sync_dist=True)
        else:
            self.log("val_loss", loss)
        self.val_acc(y_hat, y_var)
        self.log("val_acc", self.val_acc.compute())
        return {"val_step_loss": loss, "val_loss": loss}

    def configure_optimizers(self):
        """Initializes the optimizer and learning rate scheduler.
        Returns:
             output - Initialized optimizer and scheduler
        """
        if str(dict_args["deepspeed"]).lower() != "false":
            from deepspeed.ops.adam import DeepSpeedCPUAdam

            return DeepSpeedCPUAdam(self.parameters())
        else:
            self.optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.args.get("lr", 0.001),
                betas=self.args.get("betas", [0.8, 0.999]),
                weight_decay=self.args.get("weight_decay", 3e-7),
                eps=self.args.get("eps", 1e-8),
            )
            self.scheduler = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode="min",
                    factor=0.2,
                    patience=3,
                    min_lr=1e-6,
                    verbose=True,
                ),
                "monitor": "val_loss",
            }
            return [self.optimizer], [self.scheduler]

    def makegrid(self, output, numrows):  # pylint: disable=no-self-use
        """Makes grids.
        Args:
             output : Tensor output
             numrows : num of rows.
        Returns:
             c_array : gird array
        """
        outer = torch.Tensor.cpu(output).detach()
        plt.figure(figsize=(20, 5))
        b_array = np.array([]).reshape(0, outer.shape[2])
        c_array = np.array([]).reshape(numrows * outer.shape[2], 0)
        i = 0
        j = 0
        while i < outer.shape[1]:
            img = outer[0][i]
            b_array = np.concatenate((img, b_array), axis=0)
            j += 1
            if j == numrows:
                c_array = np.concatenate((c_array, b_array), axis=1)
                b_array = np.array([]).reshape(0, outer.shape[2])
                j = 0

            i += 1
        return c_array

    def show_activations(self, x_var):
        """Showns activation
        Args:
             x_var: x variable
        """

        # logging reference image
        self.logger.experiment.add_image(
            "input", torch.Tensor.cpu(x_var[0][0]), self.current_epoch, dataformats="HW"
        )

        # logging layer 1 activations
        out = self.model_conv.conv1(x_var)
        c_grid = self.makegrid(out, 4)
        self.logger.experiment.add_image("layer 1", c_grid, self.current_epoch, dataformats="HW")

    def training_epoch_end(self, outputs):
        """Training epoch end.
        Args:
             outputs: outputs of train end
        """
        self.show_activations(self.reference_image)

        # Logging graph
        if self.current_epoch == 0:
            sample_img = torch.rand((1, 3, 64, 64))
            self.logger.experiment.add_graph(CIFAR10Classifier(), sample_img)


if __name__ == "__main__":
    parser = ArgumentParser(description="PyTorch Cifar10 Example")
    parser.add_argument(
        "--download_path",
        type=str,
        default="output/processing",
        help="Path to write cifar10 dataset",
    )

    parser.add_argument(
        "--deepspeed",
        type=str,
        default="false",
        help="Enable deepspeed training",
    )

    parser = pl.Trainer.add_argparse_args(parent_parser=parser)

    from cifar10_datamodule import CIFAR10DataModule

    parser = CIFAR10DataModule.add_model_specific_args(parent_parser=parser)

    args = parser.parse_args()
    dict_args = vars(args)

    if "accelerator" in dict_args:
        if dict_args["accelerator"] == "None":
            dict_args["accelerator"] = None

    mlflow.pytorch.autolog()

    model = CIFAR10Classifier(**dict_args)

    dm = CIFAR10DataModule(**dict_args)
    dm.prepare_data()
    dm.setup(stage="fit")

    if str(dict_args["deepspeed"]).lower() != "false":
        deepspeed_config = {
            "zero_allow_untested_optimizer": True,
            "optimizer": {
                "type": "Adam",
                "params": {"lr": 0.001, "betas": [0.8, 0.999], "eps": 1e-8, "weight_decay": 3e-7},
            },
            "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "last_batch_iteration": -1,
                    "warmup_min_lr": 0,
                    "warmup_max_lr": 3e-5,
                    "warmup_num_steps": 100,
                },
            },
            "zero_optimization": {
                "stage": 2,  # Enable Stage 2 ZeRO (Optimizer/Gradient state partitioning)
                "contiguous_gradients": False,  # Reduce gradient fragmentation.
                "reduce_bucket_size": 27000000,
            },
        }

        dict_args["plugins"] = (DeepSpeedPlugin(config=deepspeed_config),)
        dict_args["precision"] = 16
    trainer = pl.Trainer.from_argparse_args(Namespace(**dict_args))

    trainer.fit(model, dm)
    trainer.test()

    torch.save(trainer.lightning_module.state_dict(), "resnet.pth")
