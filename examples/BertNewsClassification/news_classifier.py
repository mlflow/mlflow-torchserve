# pylint: disable=W0221
# pylint: disable=W0613
# pylint: disable=E1102
# pylint: disable=W0223
import argparse
import os
import shutil
from collections import defaultdict

import mlflow
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertModel, BertTokenizer, AdamW
from transformers import (
    get_linear_schedule_with_warmup,
)

print("PyTorch version: ", torch.__version__)

class_names = ["World", "Sports", "Business", "Sci/Tech"]


class NewsDataset(Dataset):
    """Ag News Dataset
    Args:
        Dataset
    """

    def __init__(self, dataset, tokenizer):
        """Performs initialization of tokenizer.
        Args:
             dataset: dataframe
             tokenizer: bert tokenizer
        """
        self.dataset = dataset
        self.max_length = 100
        self.tokenizer = tokenizer

    def __len__(self):
        """
        Returns:
             returns the number of datapoints in the dataframe
        """
        return len(self.dataset)

    def __getitem__(self, item):
        """Returns the review text and the targets of the specified item.
        Args:
             item: Index of sample review
        Returns:
             Returns the dictionary of review text,
             input ids, attention mask, targets
        """
        review = str(self.dataset[item]["text"])
        target = self.dataset[item]["label"]
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

        return {
            "review_text": review,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),  # pylint: disable=not-callable
            "targets": torch.tensor(
                target, dtype=torch.long
            ),  # pylint: disable=no-member,not-callable
        }


class NewsClassifier(nn.Module):
    def __init__(self, args):
        super(NewsClassifier, self).__init__()
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = int(os.environ["LOCAL_RANK"]) if torch.cuda.device_count() > 0 else "cpu"
        self.PRE_TRAINED_MODEL_NAME = "bert-base-uncased"
        self.args = vars(args)
        self.EPOCHS = args.max_epochs
        self.df = None
        self.tokenizer = None
        self.train_dataset = None
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

    def create_data_loader(self, dataset, tokenizer):
        """
        :param dataset: DataFrame input
        :param tokenizer: Bert tokenizer
        :return: output - Corresponding data loader for the given input
        """
        dataset = NewsDataset(
            dataset=dataset,
            tokenizer=tokenizer,
        )

        return DataLoader(
            dataset,
            batch_size=self.args.get("batch_size", 4),
            num_workers=self.args.get("num_workers", 1),
        )

    def prepare_data(self):
        """
        Creates train, valid and test dataloaders from the csv data
        """
        dataset = load_dataset("ag_news")
        num_train_samples = self.args["num_train_samples"]

        num_val_samples = int(num_train_samples * 0.1)
        num_train_samples -= num_val_samples
        self.train_dataset = dataset["train"].train_test_split(
            train_size=num_train_samples, test_size=num_val_samples
        )
        val_data = self.train_dataset["test"]
        self.train_dataset = self.train_dataset["train"]

        test_data = dataset["test"]
        num_test_samples = self.args["num_test_samples"]
        remaining = len(test_data) - num_test_samples
        test_data = dataset["train"].train_test_split(
            train_size=remaining, test_size=num_test_samples
        )["test"]

        self.tokenizer = BertTokenizer.from_pretrained(self.PRE_TRAINED_MODEL_NAME)

        self.train_data_loader = self.create_data_loader(self.train_dataset, self.tokenizer)

        self.val_data_loader = self.create_data_loader(val_data, self.tokenizer)
        self.test_data_loader = self.create_data_loader(test_data, self.tokenizer)

    def setOptimizer(self, model):
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
        model = model.to(self.device)
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


def setup(rank, world_size):
    """
    calls init process group in case of distributed training
    :param rank: local rank of gpu
    :param world_size: total number of gpus available
    """
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    """
    destroys the process group at the end of distributed training
    """
    dist.destroy_process_group()


def ddp_main(rank, world_size, args):
    """
    orchestrate training and testing process
    :param rank: Local rank of gpu
    :param world_size: Total number of gpus available
    :param args: Trainer specific arguments
    """
    if world_size > 0:
        setup(rank, world_size)
    args.rank = rank
    model = NewsClassifier(args)
    model = model.to(rank)
    model.prepare_data()
    model.setOptimizer(model)
    model.startTraining(model)
    if os.path.exists(args.model_save_path):
        shutil.rmtree(args.model_save_path)
    if rank == 0 or rank == "cpu":
        mlflow.pytorch.save_model(
            model,
            path=args.model_save_path,
            requirements_file="requirements.txt",
            extra_files=["class_mapping.json", "bert_base_uncased_vocab.txt"],
        )

    # mlflow.end_run()
    test_acc, _ = model.eval_model(model, model.test_data_loader)
    print(test_acc.item())

    if world_size > 0:
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
        "--num_train_samples",
        type=int,
        default=2000,
        metavar="N",
        help="Train num samples (default: 2000)",
    )

    parser.add_argument(
        "--num_test_samples",
        type=int,
        default=200,
        metavar="N",
        help="Train num samples (default: 200)",
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

    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    rank = int(os.environ["LOCAL_RANK"]) if torch.cuda.device_count() > 0 else "cpu"
    ddp_main(rank, WORLD_SIZE, args)
