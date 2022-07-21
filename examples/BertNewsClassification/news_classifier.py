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
import torch.distributed as dist
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
device = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else "cpu"


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


def create_data_loader(args, tokenizer, source, count):
    """
    :param args: User specific args such as batch size, num workers
    :param tokenizer: Bert tokenizer
    :param source: Iterable source
    :param count: Number of samples
    :return: output - Corresponding data loader for the given input
    """
    ds = NewsDataset(
        source=source,
        tokenizer=tokenizer,
        max_length=args.get("max_len", 100),
        num_samples=count,
    )

    return DataLoader(
        ds,
        batch_size=args.get("batch_size", 16),
        num_workers=args.get("num_workers", 3),
    )


def prepare_data(args):
    """
    Creates train, valid and test dataloaders from the csv data
    :param args: User specific args such as num samples, vocab url
    :return: Bert tokenizer and train, test and val data loaders
    """
    train_iter, test_iter = AG_NEWS()
    train_dataset = to_map_style_dataset(train_iter)
    test_dataset = to_map_style_dataset(test_iter)
    vocab_file = args.get("vocab_file", "bert_base_uncased_vocab.txt")
    vocab_url = args["vocab_url"]
    train_count = args["num_samples"]

    if not os.path.isfile(vocab_file):
        filePointer = requests.get(vocab_url, allow_redirects=True)
        if filePointer.ok:
            with open(vocab_file, "wb") as f:
                f.write(filePointer.content)
        else:
            raise RuntimeError("Error in fetching the vocab file")
    tokenizer = BertTokenizer.from_pretrained(vocab_file)

    num_train = int(len(train_dataset) * 0.95)
    train_dataset, val_dataset = random_split(
        train_dataset, [num_train, len(train_dataset) - num_train]
    )

    val_count = int(train_count / 10)
    test_count = int(train_count / 10)
    train_count = train_count - (val_count + test_count)

    print("Number of samples used for training: {}".format(train_count))
    print("Number of samples used for validation: {}".format(val_count))
    print("Number of samples used for test: {}".format(test_count))

    train_data_loader = create_data_loader(
        args=args, tokenizer=tokenizer, source=train_dataset, count=train_count
    )

    val_data_loader = create_data_loader(
        args=args, tokenizer=tokenizer, source=val_dataset, count=val_count
    )

    test_data_loader = create_data_loader(
        args=args, tokenizer=tokenizer, source=test_dataset, count=test_count
    )

    args["train_dataset_length"] = train_count

    return tokenizer, train_data_loader, val_data_loader, test_data_loader


class NewsClassifier(nn.Module):
    def __init__(self):
        super(NewsClassifier, self).__init__()
        self.PRE_TRAINED_MODEL_NAME = "bert-base-uncased"
        n_classes = len(class_names)
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


def setOptimizer(args, model):
    """
    Sets the optimizer and scheduler functions
    :param args: User specific args such as train length samples and epoch count
    :param model: Instance of the NewsClassifier class
    :result: instance of optimizer, scheduler and loss function
    """
    optimizer = AdamW(model.parameters(), lr=1e-3, correct_bias=False)
    total_steps = args["train_dataset_length"] * args["max_epochs"]

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    loss_fn = nn.CrossEntropyLoss().to(device)

    return optimizer, scheduler, loss_fn


def train_epoch(args, model, train_data_loader, loss_fn, optimizer, scheduler):
    """
    Training process happens and accuracy is returned as output
    :param args: User specific args such as train length samples and epoch count
    :param model: Instance of the NewsClassifier class
    :param train_data_loader: Data loader
    :param loss_fn: Instance of loss function
    :param optimizer: Instance of optimizer
    :param scheduler: Instance of scheduler
    :result: output - Accuracy of the model after training
    """

    model = model.train()
    losses = []
    correct_predictions = 0
    iterations = 0
    for data in tqdm(train_data_loader):
        iterations += 1
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        targets = data["targets"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return (
        correct_predictions.double() / iterations / args.get("batch_size", 16),
        np.mean(losses),
    )


def eval_model(args, model, val_data_loader, loss_fn):
    """
    Validation process happens and validation / test accuracy is returned as output
    :param args: User specific args such as train length samples and epoch count
    :param model: Instance of the NewsClassifier class
    :param val_data_loader: Data loader for either test / validation dataset
    :param loss_fn: Instance of loss function
    :result: output - Accuracy of the model after testing
    """
    model = model.eval()

    losses = []
    correct_predictions = 0
    iterations = 0

    with torch.no_grad():
        for d in val_data_loader:
            iterations += 1
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    return correct_predictions.double() / iterations / args.get("batch_size", 16), np.mean(losses)


def startTraining(args, model, train_data_loader, val_data_loader, loss_fn, optimizer, scheduler):
    """
    Initialzes the Traning step with the model initialized
    :param args: User specific args such as train length samples and epoch count
    :param model: Instance of the NewsClassifier class
    :param train_data_loader: Training Data loader
    :param val_data_loader: Validation Data loader
    :param loss_fn: Instance of loss function
    :param optimizer: Instance of optimizer
    :param scheduler: Instance of scheduler
    """
    history = defaultdict(list)
    best_accuracy = 0

    for epoch in range(args["max_epochs"]):

        print(f"Epoch {epoch + 1}/{args['max_epochs']}")

        train_acc, train_loss = train_epoch(
            args, model, train_data_loader, loss_fn, optimizer, scheduler
        )

        print(f"Train loss {train_loss} accuracy {train_acc}")

        val_acc, val_loss = eval_model(args, model, val_data_loader, loss_fn)
        print(f"Val   loss {val_loss} accuracy {val_acc}")

        history["train_acc"].append(train_acc)
        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        history["val_loss"].append(val_loss)

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), "best_model_state.bin")
            best_accuracy = val_acc


def get_predictions(model, test_data_loader):

    """
    Prediction after the training step is over
    :param model: Instance of the NewsClassifier class
    :param test_data_loader: Data loader for either test / validation dataset
    :result: output - Returns prediction results,
                      prediction probablities and corresponding values
    """
    model = model.eval()

    review_texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in test_data_loader:
            texts = d["review_text"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

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


def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def ddp_main(rank, world_size, args):
    """
    Orchestrates the training, validation, testing process and saves the model
    :param model: Instance of the NewsClassifier class
    """
    if device != "cpu":
        setup(rank, world_size)
    mlflow.start_run()

    tokenizer, train_data_loader, val_data_loader, test_data_loader = prepare_data(args=args)

    model = NewsClassifier()

    model = model.to(device)

    optimizer, scheduler, loss_fn = setOptimizer(args, model)

    startTraining(
        args=args,
        model=model,
        train_data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    test_acc, _ = eval_model(
        args=args, model=model, val_data_loader=val_data_loader, loss_fn=loss_fn
    )

    print(test_acc.item())

    get_predictions(model=model, test_data_loader=test_data_loader)

    if device == "cpu" or rank == 0:
        print("\n\n\n SAVING MODEL")
        if os.path.exists(args["model_save_path"]):
            shutil.rmtree(args["model_save_path"])

        mlflow.pytorch.save_model(
            model,
            path=args["model_save_path"],
            pip_requirements="requirements.txt",
            extra_files=["class_mapping.json", "bert_base_uncased_vocab.txt"],
        )

    mlflow.end_run()

    if device != "cpu":
        cleanup()


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
        help="Number of samples to be used for training " "and evaluation steps (default: 2000)",
    )

    parser.add_argument(
        "--batch_size", type=int, default=16, metavar="N", help="Training batch size (default: 16)"
    )

    parser.add_argument(
        "--vocab_url",
        default="https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
        help="Custom vocab file",
    )

    parser.add_argument(
        "--model_save_path", type=str, default="models", help="Path to save mlflow model"
    )

    args = parser.parse_args()
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    # When gpus are available
    if torch.cuda.device_count() > 0:
        if "LOCAL_RANK" in os.environ:
            rank = int(os.environ["LOCAL_RANK"])
        else:
            rank = "cpu"

    else:
        # No gpus available . Set the rank to cpu
        rank = "cpu"
    # rank = int(os.environ.get("LOCAL_RANK", "cpu")) if torch.cuda.device_count() > 0 else "cpu"
    ddp_main(rank, WORLD_SIZE, vars(args))
