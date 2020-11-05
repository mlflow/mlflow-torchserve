import mlflow
import os
import re
import pytest
from mlflow.utils import process

EXAMPLES_DIR = "examples"


def is_conda_yaml(path):
    return bool(re.search("conda.ya?ml$", path))


def find_conda_yaml(directory):
    conda_yaml = list(filter(is_conda_yaml, os.listdir(directory)))[0]
    return os.path.join(directory, conda_yaml)


def replace_mlflow_with_dev_version(yml_path):
    with open(yml_path, "r") as f:
        old_src = f.read()
        mlflow_dir = os.path.dirname(mlflow.__path__[0])
        new_src = re.sub(r"- mlflow.*\n", "- {}\n".format(mlflow_dir), old_src)

    with open(yml_path, "w") as f:
        f.write(new_src)


@pytest.mark.large
@pytest.mark.parametrize(
    "directory, command",
    [
        ("MNIST/example1", ["python", "mnist_model.py", "--epochs", "1"]),
        ("MNIST/example2", ["python", "mnist_model.py", "--epochs", "1"]),
        ("MNIST/example3", ["python", "mnist_model.py", "--epochs", "1"]),
        (
            "BertNewsClassification",
            ["python", "news_classifier.py", "--epochs", "1", "--num-samples", "200"],
        ),
    ],
)
def test_command_example(directory, command):
    cwd_dir = os.path.join(EXAMPLES_DIR, directory)
    process.exec_cmd(command, cwd=cwd_dir)
