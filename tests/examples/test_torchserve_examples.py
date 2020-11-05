import mlflow
import os
import re
import pytest
import shutil
from mlflow.utils import process
from mlflow import cli
from click.testing import CliRunner

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
    "directory, params",
    [
        ("MNIST/example1", ["-P", "epochs=1"]),
        ("MNIST/example2", ["-P", "epochs=1"]),
        ("MNIST/example3", ["-P", "epochs=1"]),
        ("BertNewsClassification", ["-P", "epochs=1", "-P", "num_samples=100"]),
    ],
)
def test_mlflow_run_example(directory, params, tmpdir):
    example_dir = os.path.join(EXAMPLES_DIR, directory)
    tmp_example_dir = os.path.join(tmpdir.strpath, directory)

    shutil.copytree(example_dir, tmp_example_dir)
    conda_yml_path = find_conda_yaml(tmp_example_dir)
    replace_mlflow_with_dev_version(conda_yml_path)

    cli_run_list = [tmp_example_dir] + params
    res = CliRunner().invoke(cli.run, cli_run_list)
    assert res.exit_code == 0, "Got non-zero exit code {0}. Output is: {1}".format(
        res.exit_code, res.output
    )


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
