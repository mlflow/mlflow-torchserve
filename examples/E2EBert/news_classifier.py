# pylint: disable=arguments-differ
# pylint: disable=unused-argument
# pylint: disable=abstract-method

import math
import os

import lightning as L
import mlflow.pytorch
import numpy as np
import requests
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from lightning.pytorch.cli import LightningCLI
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchdata.datapipes.iter import IterDataPipe
from torchtext.data.functional import to_map_style_dataset
from torchtext.datasets import AG_NEWS
from transformers import BertModel, BertTokenizer, AdamW


class NewsDataset(IterDataPipe):
    def __init__(self, tokenizer, source, max_length, num_samples):
        """
        Custom Dataset - Converts the input text and label to tensor
        :param tokenizer: bert tokenizer
        :param source: data source - Either a dataframe or DataPipe
        :param max_length: maximum length of the news text
        :param num_samples: number of samples to load
        :param dataset: Dataset type - 20newsgroups or ag_news
        """
        super(NewsDataset, self).__init__()
        self.source = source
        self.start = 0
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.end = num_samples

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
            target, review = self.source[idx]
            target -= 1
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

            yield {
                "review_text": review,
                "input_ids": encoding["input_ids"].flatten(),
                "attention_mask": encoding["attention_mask"].flatten(),
                "targets": torch.tensor(target, dtype=torch.long),
            }


class BertDataModule(L.LightningDataModule):
    def __init__(self, batch_size=4, num_workers=3, num_samples=2000):
        """
        Initialization of inherited lightning data module
        """
        super(BertDataModule, self).__init__()
        self.PRE_TRAINED_MODEL_NAME = "bert-base-uncased"
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.MAX_LEN = 100
        self.encoding = None
        self.tokenizer = None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_samples = num_samples
        self.train_count = None
        self.val_count = None
        self.test_count = None
        self.RANDOM_SEED = 42
        self.VOCAB_FILE_URL = (
            "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt"
        )
        self.VOCAB_FILE = "bert_base_uncased_vocab.txt"

    def prepare_data(self):
        """
        Downloads the ag_news or 20newsgroup dataset and initializes bert tokenizer
        """
        np.random.seed(self.RANDOM_SEED)
        torch.manual_seed(self.RANDOM_SEED)

        train_iter, test_iter = AG_NEWS()
        self.train_dataset = to_map_style_dataset(train_iter)
        self.test_dataset = to_map_style_dataset(test_iter)

        if not os.path.isfile(self.VOCAB_FILE):
            filePointer = requests.get(self.VOCAB_FILE_URL, allow_redirects=True)
            if filePointer.ok:
                with open(self.VOCAB_FILE, "wb") as f:
                    f.write(filePointer.content)
            else:
                raise RuntimeError("Error in fetching the vocab file")
        self.tokenizer = BertTokenizer.from_pretrained(self.VOCAB_FILE)

    def setup(self, stage=None):
        """
        Split the data into train, test, validation data
        :param stage: Stage - training or testing
        """
        if stage == "fit":

            num_train = int(len(self.train_dataset) * 0.95)
            self.train_dataset, self.val_dataset = random_split(
                self.train_dataset, [num_train, len(self.train_dataset) - num_train]
            )

            self.train_count = self.num_samples
            self.val_count = int(self.train_count / 10)
            self.test_count = int(self.train_count / 10)
            self.train_count = self.train_count - (self.val_count + self.test_count)

            print("Number of samples used for training: {}".format(self.train_count))
            print("Number of samples used for validation: {}".format(self.val_count))
            print("Number of samples used for test: {}".format(self.test_count))

    def create_data_loader(self, source, count):
        """
        Generic data loader function
        :param df: Input dataframe
        :param tokenizer: bert tokenizer
        :return: Returns the constructed dataloader
        """
        ds = NewsDataset(
            source=source,
            tokenizer=self.tokenizer,
            max_length=self.MAX_LEN,
            num_samples=count,
        )

        return DataLoader(ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def train_dataloader(self):
        """
        :return: output - Train data loader for the given input
        """
        return self.create_data_loader(source=self.train_dataset, count=self.train_count)

    def val_dataloader(self):
        """
        :return: output - Validation data loader for the given input
        """
        return self.create_data_loader(source=self.val_dataset, count=self.val_count)

    def test_dataloader(self):
        """
        :return: output - Test data loader for the given input
        """
        return self.create_data_loader(source=self.test_dataset, count=self.test_count)


class BertNewsClassifier(L.LightningModule):
    def __init__(self, learning_rate=0.001):
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
        self.learning_rate = learning_rate
        self.val_outputs = []
        self.test_outputs = []

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
        self.val_outputs.append(loss)
        return {"val_step_loss": loss}

    def on_validation_epoch_end(self):
        """
        Computes average validation accuracy
        """
        avg_loss = torch.stack(self.val_outputs).mean()
        self.log("val_loss", avg_loss, sync_dist=True)
        self.val_outputs.clear()

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
        self.test_outputs.append(torch.tensor(test_acc))
        return {"test_acc": torch.tensor(test_acc)}

    def on_test_epoch_end(self):
        """
        Computes average test accuracy score
        """
        print(self.test_outputs)
        avg_test_acc = torch.stack(self.test_outputs).mean()
        self.log("avg_test_acc", avg_test_acc)
        self.test_outputs.clear()

    def configure_optimizers(self):
        """
        Initializes the optimizer and learning rate scheduler
        :return: output - Initialized optimizer and scheduler
        """
        self.optimizer = AdamW(self.parameters(), lr=self.learning_rate)
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


def cli_main():
    early_stopping = EarlyStopping(
        monitor="val_loss",
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.getcwd(), save_top_k=1, verbose=True, monitor="val_loss", mode="min"
    )
    lr_logger = LearningRateMonitor()
    cli = LightningCLI(
        BertNewsClassifier,
        BertDataModule,
        run=False,
        save_config_callback=None,
        trainer_defaults={"callbacks": [early_stopping, checkpoint_callback, lr_logger]},
    )
    if cli.trainer.global_rank == 0:
        mlflow.pytorch.autolog()
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)

    if cli.trainer.global_rank == 0:
        torch.save(cli.model.state_dict(), "state_dict.pth")


if __name__ == "__main__":
    cli_main()
