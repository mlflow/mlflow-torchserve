# pylint: disable=W0223
import argparse
import json
import mlflow
import mlflow.pytorch
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self):
        """
        Initializes the layers of the model
        """
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """
        Does the propagation of the network with the inputs given

        :param x: Input to be passed through the layers for prediction

        :return: output - Output of the softmax layer
        """
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class BaseMNIST(object):
    def __init__(self):
        pass

    def train(self, args, model, device, train_loader, optimizer, epoch):
        """
        Training the data as batches

        :param args: Input cli arguments
        :param model: Instance of the class Net
        :param device: CUDA support if available
        :param train_loader: Dataloader with training data
        :param optimizer: optimzier to be used in the training of the model
        :param epoch: Number of epochs

        """

        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )

    def test(self, model, device, test_loader):
        """
        Tesing the model after training has been done on unseen data

        :param model: Instance of the class Net
        :param device: CUDA support if available
        :param test_loader: Dataloader with testing data

        """
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss,
                correct,
                len(test_loader.dataset),
                100.0 * correct / len(test_loader.dataset),
            )
        )


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example1")
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--model-save-path",
        type=str,
        default="models",
        help="Directory path for saving the model file",
    )
    parser.add_argument(
        "--generate-sample-input",
        type=bool,
        default=True,
        help="Creates Sample input file for deployment",
    )
    parser.add_argument(
        "--save-mlflow-model",
        type=bool,
        default=False,
        help="Save model using mlflow.pytoch library",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kwargs = {"shuffle": True, "batch_size": 64}

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset1 = datasets.MNIST("../data", train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST("../data", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=0.01)

    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    base_mnist = BaseMNIST()
    for epoch in range(1, args.epochs + 1):
        base_mnist.train(args, model, device, train_loader, optimizer, epoch)
        base_mnist.test(model, device, test_loader)
        scheduler.step()

    if args.save_mlflow_model:
        with mlflow.start_run():
            mlflow.pytorch.save_model(
                model,
                path=os.path.join(args.model_save_path, "models"),
                requirements_file="requirements.txt",
                extra_files=["number_to_text.json"],
            )
    else:
        torch.save(model.state_dict(), os.path.join(args.model_save_path, "mnist_cnn.pt"))

    if args.generate_sample_input:
        model_dir = os.path.dirname(os.path.abspath(__file__))
        data = {"data": [os.path.join(model_dir, "one.png")]}
        with open(os.path.join(args.model_save_path, "sample.json"), "w") as fp:
            json.dump(data, fp)


if __name__ == "__main__":
    main()
