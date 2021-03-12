import os
import pytest
from mlflow import cli
from click.testing import CliRunner
import shutil
from mlflow.utils import process

EXAMPLES_DIR = "examples"


@pytest.mark.parametrize(
    "directory, params",
    [
        ("IrisClassification", ["-P", "max_epochs=10"]),
        ("MNIST", ["-P", "max_epochs=1"]),
        ("IrisClassificationTorchScript", ["-P", "max_epochs=10"]),
        ("BertNewsClassification", ["-P", "max_epochs=1", "-P", "num_samples=100"]),
        ("E2EBert", ["-P", "max_epochs=1", "-P", "num_samples=100"]),
        ("Titanic", ["-P", "max_epochs=100", "-P", "lr=0.1"]),
    ],
)
def test_mlflow_run_example(directory, params):
    example_dir = os.path.join(EXAMPLES_DIR, directory)
    cli_run_list = [example_dir] + params
    res = CliRunner().invoke(cli.run, cli_run_list)
    assert res.exit_code == 0, "Got non-zero exit code {0}. Output is: {1}".format(
        res.exit_code, res.output
    )


def get_free_disk_space():
    # https://stackoverflow.com/a/48929832/6943581
    return shutil.disk_usage("/")[-1] / (2 ** 30)


@pytest.fixture(scope="function", autouse=True)
def clean_envs_and_cache():
    yield

    if get_free_disk_space() < 7.0:  # unit: GiB
        process.exec_cmd(["./utils/remove-conda-envs.sh"])
