# pylint: disable=W0221
# pylint: disable=W0613
# pylint: disable=E1102
# pylint: disable=W0223
import argparse
import math
import os
import shutil
from collections import defaultdict

import mlflow.pytorch
import numpy as np
import requests
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchdata.datapipes.iter import IterDataPipe
from torchtext.data.functional import to_map_style_dataset
from torchtext.datasets import AG_NEWS
from tqdm import tqdm
from transformers import BertModel, BertTokenizer, AdamW
from transformers import (
    get_linear_schedule_with_warmup,
)

class_names = ["World", "Sports", "Business", "Sci/Tech"]


class NewsDataset(IterDataPipe):
    def __init__(self, tokenizer, source, max_length, num_samples):
        """
        Custom Dataset - Converts the input text and label to tensor
        :param tokenizer: bert tokenizer
        :param source: data source - Either a dataframe or DataPipe
        :param max_length: maximum length of the news text
        :param num_samples: number of samples to load
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


class NewsClassifier(nn.Module):
    def __init__(self, args):
        super(NewsClassifier, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.PRE_TRAINED_MODEL_NAME = "bert-base-uncased"
        self.args = vars(args)
        self.EPOCHS = args.max_epochs
        self.df = None
        self.tokenizer = None
        self.df_train = None
        self.df_val = None
        self.df_test = None
        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None
        self.optimizer = None
        self.total_steps = None
        self.scheduler = None
        self.loss_fn = None
        self.BATCH_SIZE = 16
        self.MAX_LEN = 160
        n_classes = len(class_names)
        self.VOCAB_FILE_URL = args.vocab_file
        self.VOCAB_FILE = "bert_base_uncased_vocab.txt"
        self.train_dataset = None

        self.drop = nn.Dropout(p=0.2)
        self.bert = BertModel.from_pretrained(self.PRE_TRAINED_MODEL_NAME)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 512)
        self.out = nn.Linear(512, n_classes)

    def forward(self, input_ids, attention_mask):
        """
        :param input_ids: Input sentences from the batch
        :param attention_mask: Attention mask returned by the encoder

        :return: output - label for the input text
        """
        pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        output = F.relu(self.fc1(pooled_output))
        output = self.drop(output)
        output = self.out(output)
        return output

    @staticmethod
    def process_label(rating):
        rating = int(rating)
        return rating - 1

    def create_data_loader(self, source, count):
        """
        :param source: Iterable source
        :param count: Number of samples

        :return: output - Corresponding data loader for the given input
        """
        ds = NewsDataset(
            source=source,
            tokenizer=self.tokenizer,
            max_length=self.MAX_LEN,
            num_samples=count,
        )

        return DataLoader(
            ds,
            batch_size=self.args.get("batch_size", self.BATCH_SIZE),
            num_workers=self.args.get("num_workers", 3),
        )

    def prepare_data(self):
        """
        Creates train, valid and test dataloaders from the csv data
        """
        train_iter, test_iter = AG_NEWS()
        self.train_dataset = to_map_style_dataset(train_iter)
        test_dataset = to_map_style_dataset(test_iter)

        if not os.path.isfile(self.VOCAB_FILE):
            filePointer = requests.get(self.VOCAB_FILE_URL, allow_redirects=True)
            if filePointer.ok:
                with open(self.VOCAB_FILE, "wb") as f:
                    f.write(filePointer.content)
            else:
                raise RuntimeError("Error in fetching the vocab file")
        self.tokenizer = BertTokenizer.from_pretrained(self.VOCAB_FILE)

        num_train = int(len(self.train_dataset) * 0.95)
        self.train_dataset, val_dataset = random_split(
            self.train_dataset, [num_train, len(self.train_dataset) - num_train]
        )

        train_count = self.args.get("num_samples")
        val_count = int(train_count / 10)
        test_count = int(train_count / 10)
        train_count = train_count - (val_count + test_count)

        print("Number of samples used for training: {}".format(train_count))
        print("Number of samples used for validation: {}".format(val_count))
        print("Number of samples used for test: {}".format(test_count))

        self.train_data_loader = self.create_data_loader(
            source=self.train_dataset, count=train_count
        )

        self.val_data_loader = self.create_data_loader(source=val_dataset, count=val_count)

        self.test_data_loader = self.create_data_loader(source=test_dataset, count=test_count)

    def setOptimizer(self):
        """
        Sets the optimizer and scheduler functions
        """
        self.optimizer = AdamW(model.parameters(), lr=1e-3, correct_bias=False)
        self.total_steps = len(self.train_dataset) * self.EPOCHS

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=0, num_training_steps=self.total_steps
        )

        self.loss_fn = nn.CrossEntropyLoss().to(self.device)

    def startTraining(self, model):
        """
        Initialzes the Traning step with the model initialized

        :param model: Instance of the NewsClassifier class
        """
        history = defaultdict(list)
        best_accuracy = 0

        for epoch in range(self.EPOCHS):

            print(f"Epoch {epoch + 1}/{self.EPOCHS}")

            train_acc, train_loss = self.train_epoch(model)

            print(f"Train loss {train_loss} accuracy {train_acc}")

            val_acc, val_loss = self.eval_model(model, self.val_data_loader)
            print(f"Val   loss {val_loss} accuracy {val_acc}")

            history["train_acc"].append(train_acc)
            history["train_loss"].append(train_loss)
            history["val_acc"].append(val_acc)
            history["val_loss"].append(val_loss)

            if val_acc > best_accuracy:
                torch.save(model.state_dict(), "best_model_state.bin")
                best_accuracy = val_acc

    def train_epoch(self, model):
        """
        Training process happens and accuracy is returned as output

        :param model: Instance of the NewsClassifier class

        :result: output - Accuracy of the model after training
        """

        model = model.train()
        losses = []
        correct_predictions = 0
        iterations = 0
        for data in tqdm(self.train_data_loader):
            iterations += 1
            input_ids = data["input_ids"].to(self.device)
            attention_mask = data["attention_mask"].to(self.device)
            targets = data["targets"].to(self.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            _, preds = torch.max(outputs, dim=1)
            loss = self.loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        return (
            correct_predictions.double() / iterations / self.BATCH_SIZE,
            np.mean(losses),
        )

    def eval_model(self, model, data_loader):
        """
        Validation process happens and validation / test accuracy is returned as output

        :param model: Instance of the NewsClassifier class
        :param data_loader: Data loader for either test / validation dataset

        :result: output - Accuracy of the model after testing
        """
        model = model.eval()

        losses = []
        correct_predictions = 0
        iterations = 0

        with torch.no_grad():
            for d in data_loader:
                iterations += 1
                input_ids = d["input_ids"].to(self.device)
                attention_mask = d["attention_mask"].to(self.device)
                targets = d["targets"].to(self.device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs, dim=1)

                loss = self.loss_fn(outputs, targets)
                correct_predictions += torch.sum(preds == targets)
                losses.append(loss.item())

        return correct_predictions.double() / iterations / self.BATCH_SIZE, np.mean(losses)

    def get_predictions(self, model, data_loader):

        """
        Prediction after the training step is over

        :param model: Instance of the NewsClassifier class
        :param data_loader: Data loader for either test / validation dataset

        :result: output - Returns prediction results,
                          prediction probablities and corresponding values
        """
        model = model.eval()

        review_texts = []
        predictions = []
        prediction_probs = []
        real_values = []

        with torch.no_grad():
            for d in data_loader:
                texts = d["review_text"]
                input_ids = d["input_ids"].to(self.device)
                attention_mask = d["attention_mask"].to(self.device)
                targets = d["targets"].to(self.device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs, dim=1)

                probs = F.softmax(outputs, dim=1)

                review_texts.extend(texts)
                predictions.extend(preds)
                prediction_probs.extend(probs)
                real_values.extend(targets)

        predictions = torch.stack(predictions).cpu()
        prediction_probs = torch.stack(prediction_probs).cpu()
        real_values = torch.stack(real_values).cpu()
        return review_texts, predictions, prediction_probs, real_values


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="PyTorch BERT Example")

    parser.add_argument(
        "--max_epochs",
        type=int,
        default=5,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=2000,
        metavar="N",
        help="Number of samples to be used for training "
        "and evaluation steps (default: 2000)",
    )

    parser.add_argument(
        "--vocab_file",
        default="https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
        help="Custom vocab file",
    )

    parser.add_argument(
        "--model_save_path", type=str, default="models", help="Path to save mlflow model"
    )

    args = parser.parse_args()
    mlflow.start_run()

    model = NewsClassifier(args)
    model = model.to(model.device)
    model.prepare_data()
    model.setOptimizer()
    model.startTraining(model)

    print("TRAINING COMPLETED!!!")

    test_acc, _ = model.eval_model(model, model.test_data_loader)

    print(test_acc.item())

    y_review_texts, y_pred, y_pred_probs, y_test = model.get_predictions(
        model, model.test_data_loader
    )

    print("\n\n\n SAVING MODEL")

    if os.path.exists(args.model_save_path):
        shutil.rmtree(args.model_save_path)
    mlflow.pytorch.save_model(
        model,
        path=args.model_save_path,
        requirements_file="requirements.txt",
        extra_files=["class_mapping.json"],
    )

    mlflow.end_run()
