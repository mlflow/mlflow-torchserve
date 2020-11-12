import os
import pytest
from mlflow.utils import process
from mlflow import cli
from click.testing import CliRunner

EXAMPLES_DIR = "examples"


@pytest.mark.parametrize(
    "directory, params",
    [
        ("IrisClassification", []),
    ],
)
def test_mlflow_run_example(directory, params):
    example_dir = os.path.join(EXAMPLES_DIR, directory)
    cli_run_list = [example_dir] + params
    res = CliRunner().invoke(cli.run, cli_run_list)
    assert res.exit_code == 0, "Got non-zero exit code {0}. Output is: {1}".format(
        res.exit_code, res.output
    )


@pytest.mark.parametrize(
    "directory, command",
    [
        ("IrisClassification", ["python", "iris_classification.py"]),
    ],
)
def test_command_example(directory, command):
    cwd_dir = os.path.join(EXAMPLES_DIR, directory)
    process.exec_cmd(command, cwd=cwd_dir)
