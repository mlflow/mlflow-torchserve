import json
import os

import pytest
from click.testing import CliRunner
from mlflow import deployments
from mlflow.deployments import cli

f_target = "torchserve"
f_deployment_id = "cli_test"
f_deployment_name_version = "cli_test/2.0"
f_deployment_name_all = "cli_test/all"
f_flavor = None
f_model_uri = os.path.join("tests/resources", "linear_state_dict.pt")

model_version = "1.0"
model_file_path = os.path.join("tests/resources", "linear_model.py")
handler_file_path = os.path.join("tests/resources", "linear_handler.py")
sample_input_file = os.path.join("tests/resources", "sample.json")


@pytest.mark.usefixtures("start_torchserve")
def test_create_cli_version_success():
    version = "VERSION={version}".format(version="1.0")
    model_file = "MODEL_FILE={model_file_path}".format(model_file_path=model_file_path)
    handler_file = "HANDLER={handler_file_path}".format(handler_file_path=handler_file_path)
    _ = deployments.get_deploy_client(f_target)
    runner = CliRunner()
    res = runner.invoke(
        cli.create_deployment,
        [
            "-f",
            f_flavor,
            "-m",
            f_model_uri,
            "-t",
            f_target,
            "--name",
            f_deployment_id,
            "-C",
            model_file,
            "-C",
            handler_file,
            "-C",
            version,
        ],
    )
    assert "{} deployment {} is created".format(f_flavor, f_deployment_id + "/1.0") in res.stdout


def test_create_cli_success_without_version():
    model_file = "MODEL_FILE={model_file_path}".format(model_file_path=model_file_path)
    handler_file = "HANDLER={handler_file_path}".format(handler_file_path=handler_file_path)
    _ = deployments.get_deploy_client(f_target)
    runner = CliRunner()
    res = runner.invoke(
        cli.create_deployment,
        [
            "-f",
            f_flavor,
            "-m",
            f_model_uri,
            "-t",
            f_target,
            "--name",
            f_deployment_id,
            "-C",
            model_file,
            "-C",
            handler_file,
        ],
    )
    assert "{} deployment {} is created".format(f_flavor, f_deployment_name_version) in res.stdout


@pytest.mark.parametrize(
    "deployment_name, config",
    [(f_deployment_name_version, "SET-DEFAULT=true"), (f_deployment_id, "MIN_WORKER=3")],
)
def test_update_cli_success(deployment_name, config):
    runner = CliRunner()
    res = runner.invoke(
        cli.update_deployment,
        [
            "--flavor",
            f_flavor,
            "--model-uri",
            f_model_uri,
            "--target",
            f_target,
            "--name",
            deployment_name,
            "-C",
            config,
        ],
    )
    assert (
        "Deployment {} is updated (with flavor {})".format(deployment_name, f_flavor) in res.stdout
    )


def test_list_cli_success():
    runner = CliRunner()
    res = runner.invoke(cli.list_deployment, ["--target", f_target])
    assert "{}".format(f_deployment_id) in res.stdout


@pytest.mark.parametrize(
    "deployment_name", [f_deployment_id, f_deployment_name_version, f_deployment_name_all]
)
def test_get_cli_success(deployment_name):
    runner = CliRunner()
    res = runner.invoke(cli.get_deployment, ["--name", deployment_name, "--target", f_target])
    ret = json.loads(res.stdout.split("deploy:")[1])
    if deployment_name == f_deployment_id:
        assert ret[0]["modelName"] == f_deployment_id
    elif deployment_name == f_deployment_name_version:
        assert ret[0]["modelVersion"] == f_deployment_name_version.split("/")[1]
    else:
        assert len(ret) == 2


@pytest.mark.parametrize("deployment_name", [f_deployment_name_version, f_deployment_id])
def test_predict_cli_success(deployment_name):
    runner = CliRunner()
    res = runner.invoke(
        cli.predict,
        ["--name", deployment_name, "--target", f_target, "--input-path", sample_input_file],
    )
    assert res.stdout != ""


@pytest.mark.parametrize("deployment_name", [f_deployment_id + "/1.0", f_deployment_name_version])
def test_delete_cli_success(deployment_name):
    runner = CliRunner()
    res = runner.invoke(
        cli.delete_deployment,
        ["--name", deployment_name, "--target", f_target],
    )
    assert "Deployment {} is deleted".format(deployment_name) in res.stdout
