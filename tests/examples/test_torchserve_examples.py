import os
import pytest
from mlflow import cli
from click.testing import CliRunner

EXAMPLES_DIR = "examples"


@pytest.mark.parametrize(
    "directory, params",
    [
        ("IrisClassification", ["-P", "max_epochs=10"]),
        ("MNIST", ["-P", "max_epochs=1"]),
        ("IrisClassificationTorchScript", ["-P", "max_epochs=10"]),
        ("BertNewsClassification", ["-P", "max_epochs=1", "-P", "num_samples=100"]),
        ("E2EBert", ["-P", "max_epochs=1", "-P", "num_samples=100"]),
    ],
)
def test_mlflow_run_example(directory, params):
    example_dir = os.path.join(EXAMPLES_DIR, directory)
    cli_run_list = [example_dir] + params
    res = CliRunner().invoke(cli.run, cli_run_list)
    assert res.exit_code == 0, "Got non-zero exit code {0}. Output is: {1}".format(
        res.exit_code, res.output
    )
