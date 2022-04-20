# pylint: disable=arguments-differ
# pylint: disable=unused-argument
# pylint: disable=abstract-method

import glob
import logging
import math
import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterDataPipe, random_split
from torchtext.data.functional import to_map_style_dataset
from torchtext.datasets import AG_NEWS
from transformers import BertModel, BertTokenizer, AdamW


class NewsDataset(IterDataPipe):
    def __init__(self, tokenizer, source_datapipe, max_length, num_samples=None):
        super(NewsDataset, self).__init__()
        self.source_datapipe = source_datapipe
        self.start = 0
        self.tokenizer = tokenizer
        self.max_length = max_length
        if num_samples:
            self.end = num_samples
        else:
            self.end = len(self.source_datapipe)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = self.start
            iter_end = self.end
        else:
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)

        for idx in range(iter_start, iter_end):
            target, review = self.source_datapipe[idx]
            # print(target, review)
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
            target -= 1

            yield {
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
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None
        self.MAX_LEN = 100
        self.encoding = None
        self.tokenizer = None
        self.args = kwargs

    def prepare_data(self):
        """
        Downloads the data, parse it and split the data into train, test, validation data
        :param stage: Stage - training or testing
        """
        train_iter, test_iter = AG_NEWS()
        self.train_dataset = to_map_style_dataset(train_iter)
        self.test_dataset = to_map_style_dataset(test_iter)

        num_train = int(len(self.train_dataset) * 0.95)
        self.train_dataset, self.val_dataset = random_split(
            self.train_dataset, [num_train, len(self.train_dataset) - num_train]
        )

        logging.debug("Total train samples: {}".format(len(self.train_dataset)))
        logging.debug("Total validation samples: {}".format(len(self.val_dataset)))
        logging.debug("Total test samples: {}".format(len(self.test_dataset)))
        logging.debug(
            "Number of samples to be used for training: {}".format(
                self.args.get("train_num_samples", None)
            )
        )
        logging.debug(
            "Number of samples to be used for validation: {}".format(
                self.args.get("val_num_samples", None)
            )
        )
        logging.debug(
            "Number of samples to be used for test: {}".format(
                self.args.get("test_num_samples", None)
            )
        )
        self.tokenizer = BertTokenizer.from_pretrained(self.PRE_TRAINED_MODEL_NAME)

    def setup(self, stage=None):
        pass

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Returns the review text and the targets of the specified item
        :param parent_parser: Application specific parser
        :return: Returns the augmented arugument parser
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--batch_size",
            type=int,
            default=16,
            metavar="N",
            help="input batch size for training (default: 16)",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=3,
            metavar="N",
            help="number of workers (default: 3)",
        )
        return parser

    def create_data_loader(self, source, tokenizer, max_len, batch_size, num_samples=None):
        """
        Generic data loader function
        :param df: Input dataframe
        :param tokenizer: bert tokenizer
        :param max_len: Max length of the news datapoint
        :param batch_size: Batch size for training
        :return: Returns the constructed dataloader
        """
        ds = NewsDataset(
            source_datapipe=source,
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
        ds = NewsDataset(
            source_datapipe=self.train_dataset,
            tokenizer=self.tokenizer,
            max_length=self.MAX_LEN,
            num_samples=self.args.get("train_num_samples", None),
        )

        self.train_data_loader = DataLoader(
            ds, batch_size=self.args["batch_size"], num_workers=self.args["num_workers"]
        )
        return self.train_data_loader

    def val_dataloader(self):
        """
        :return: output - Validation data loader for the given input
        """
        ds = NewsDataset(
            source_datapipe=self.val_dataset,
            tokenizer=self.tokenizer,
            max_length=self.MAX_LEN,
            num_samples=self.args.get("val_num_samples", None),
        )

        self.val_data_loader = DataLoader(
            ds, batch_size=self.args["batch_size"], num_workers=self.args["num_workers"]
        )
        return self.val_data_loader

    def test_dataloader(self):
        """
        :return: output - Test data loader for the given input
        """
        ds = NewsDataset(
            source_datapipe=self.test_dataset,
            tokenizer=self.tokenizer,
            max_length=self.MAX_LEN,
            num_samples=self.args.get("test_num_samples", None),
        )
        self.test_data_loader = DataLoader(
            ds, batch_size=self.args["batch_size"], num_workers=self.args["num_workers"]
        )
        return self.test_data_loader


class BertNewsClassifier(pl.LightningModule):
    def __init__(self, **kwargs):
        """
        Initializes the network, optimizer and scheduler
        """
        super(BertNewsClassifier, self).__init__()
        self.PRE_TRAINED_MODEL_NAME = "bert-base-uncased"
        self.bert_model = BertModel.from_pretrained(self.PRE_TRAINED_MODEL_NAME)
        for param in self.bert_model.parameters():
            param.requires_grad = False
        self.drop = nn.Dropout(p=0.2)
        # assigning labels
        self.class_names = ["world", "Sports", "Business", "Sci/Tech"]
        n_classes = len(self.class_names)

        self.fc1 = nn.Linear(self.bert_model.config.hidden_size, 512)
        self.out = nn.Linear(512, n_classes)

        self.scheduler = None
        self.optimizer = None
        self.args = kwargs

    def compute_bert_outputs(
        self, model_bert, embedding_input, attention_mask=None, head_mask=None
    ):
        if attention_mask is None:
            attention_mask = torch.ones(embedding_input.shape[0], embedding_input.shape[1]).to(
                embedding_input
            )

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(
            dtype=next(model_bert.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(model_bert.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(model_bert.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * model_bert.config.num_hidden_layers

        encoder_outputs = model_bert.encoder(
            embedding_input, extended_attention_mask, head_mask=head_mask
        )
        sequence_output = encoder_outputs[0]
        pooled_output = model_bert.pooler(sequence_output)
        outputs = (
            sequence_output,
            pooled_output,
        ) + encoder_outputs[1:]
        return outputs

    def forward(self, input_ids, attention_mask=None):
        """
        :param input_ids: Input data
        :param attention_maks: Attention mask value

        :return: output - Type of news for the given news snippet
        """
        embedding_input = self.bert_model.embeddings(input_ids)
        outputs = self.compute_bert_outputs(self.bert_model, embedding_input, attention_mask)
        pooled_output = outputs[1]
        output = F.relu(self.fc1(pooled_output))
        output = self.drop(output)
        output = self.out(output)
        return output

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Returns the review text and the targets of the specified item
        :param parent_parser: Application specific parser
        :return: Returns the augmented arugument parser
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--lr",
            type=float,
            default=0.001,
            metavar="LR",
            help="learning rate (default: 0.001)",
        )
        return parser

    def training_step(self, train_batch, batch_idx):
        """
        Training the data as batches and returns training loss on each batch
        :param train_batch Batch data
        :param batch_idx: Batch indices
        :return: output - Training loss
        """
        input_ids = train_batch["input_ids"].to(self.device)
        attention_mask = train_batch["attention_mask"].to(self.device)
        targets = train_batch["targets"].to(self.device)
        output = self.forward(input_ids, attention_mask)
        loss = F.cross_entropy(output, targets)
        self.log("train_loss", loss)
        return {"loss": loss}

    def test_step(self, test_batch, batch_idx):
        """
        Performs test and computes the accuracy of the model
        :param test_batch: Batch data
        :param batch_idx: Batch indices
        :return: output - Testing accuracy
        """
        input_ids = test_batch["input_ids"].to(self.device)
        attention_mask = test_batch["attention_mask"].to(self.device)
        targets = test_batch["targets"].to(self.device)
        output = self.forward(input_ids, attention_mask)
        _, y_hat = torch.max(output, dim=1)
        test_acc = accuracy_score(y_hat.cpu(), targets.cpu())
        return {"test_acc": torch.tensor(test_acc)}

    def validation_step(self, val_batch, batch_idx):
        """
        Performs validation of data in batches
        :param val_batch: Batch data
        :param batch_idx: Batch indices
        :return: output - valid step loss
        """

        input_ids = val_batch["input_ids"].to(self.device)
        attention_mask = val_batch["attention_mask"].to(self.device)
        targets = val_batch["targets"].to(self.device)
        output = self.forward(input_ids, attention_mask)
        loss = F.cross_entropy(output, targets)
        return {"val_step_loss": loss}

    def validation_epoch_end(self, outputs):
        """
        Computes average validation accuracy
        :param outputs: outputs after every epoch end
        :return: output - average valid loss
        """
        avg_loss = torch.stack([x["val_step_loss"] for x in outputs]).mean()
        self.log("val_loss", avg_loss, sync_dist=True)

    def test_epoch_end(self, outputs):
        """
        Computes average test accuracy score
        :param outputs: outputs after every epoch end
        :return: output - average test loss
        """
        avg_test_acc = torch.stack([x["test_acc"] for x in outputs]).mean()
        self.log("avg_test_acc", avg_test_acc)

    def configure_optimizers(self):
        """
        Initializes the optimizer and learning rate scheduler
        :return: output - Initialized optimizer and scheduler
        """
        self.optimizer = AdamW(self.parameters(), lr=self.args["lr"])
        self.scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.2,
                patience=2,
                min_lr=1e-6,
                verbose=True,
            ),
            "monitor": "val_loss",
        }
        return [self.optimizer], [self.scheduler]


if __name__ == "__main__":
    parser = ArgumentParser(description="Bert-News Classifier Example")

    parser.add_argument(
        "--train_num_samples",
        type=int,
        default=2000,
        metavar="N",
        help="Number of samples to be used for training",
    )
    parser.add_argument(
        "--val_num_samples",
        type=int,
        default=200,
        metavar="N",
        help="Number of samples to be used for validation",
    )
    parser.add_argument(
        "--test_num_samples",
        type=int,
        default=200,
        metavar="N",
        help="Number of samples to be used for testing",
    )

    parser.add_argument(
        "--save_every_n_epoch", default=5, type=int, help="Number of epochs between checkpoints"
    )

    parser.add_argument(
        "--resume",
        default=False,
        type=bool,
        help="Set to True for resuming from previous checkpoint",
    )

    parser = pl.Trainer.add_argparse_args(parent_parser=parser)
    parser = BertNewsClassifier.add_model_specific_args(parent_parser=parser)
    parser = BertDataModule.add_model_specific_args(parent_parser=parser)

    args = parser.parse_args()
    dict_args = vars(args)

    if "accelerator" in dict_args:
        if dict_args["accelerator"] == "None":
            dict_args["accelerator"] = None

    dm = BertDataModule(**dict_args)
    dm.prepare_data()

    model = BertNewsClassifier(**dict_args)
    early_stopping = EarlyStopping(monitor="val_loss", mode="min", verbose=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.getcwd(),
        every_n_epochs=dict_args["save_every_n_epoch"],
    )
    lr_logger = LearningRateMonitor()

    resume_from_checkpoint = False

    if dict_args["resume"]:
        resume_from_checkpoint = True

    if resume_from_checkpoint:
        checkpoint_list = glob.glob("epoch=*-step=*.ckpt")
        if len(checkpoint_list) == 0:
            raise Exception("Checkpoint doesn't exist")
        else:
            trainer = pl.Trainer.from_argparse_args(
                args,
                callbacks=[lr_logger, early_stopping, checkpoint_callback],
                resume_from_checkpoint=checkpoint_list[0],
                checkpoint_callback=True,
            )
    else:
        trainer = pl.Trainer.from_argparse_args(
            args,
            callbacks=[lr_logger, early_stopping, checkpoint_callback],
            checkpoint_callback=True,
        )

    trainer.fit(model, dm)
    trainer.test(model, dm)
    if trainer.global_rank == 0:
        torch.save(model.state_dict(), "state_dict.pth")
