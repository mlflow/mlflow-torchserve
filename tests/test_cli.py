import atexit
import json
import os
import shutil
import subprocess
import time

import pytest
from click.testing import CliRunner

from mlflow import deployments
from mlflow.deployments import cli

f_target = "torchserve"
f_deployment_id = "cli_test"
f_flavor = None
f_model_uri = os.path.join("mlflow/pytorch/torchserve/tests/resources", "linear.pt")

model_version = "1.0"
model_file_path = os.path.join(
    "mlflow/pytorch/torchserve/tests/resources", "linear_model.py"
)
handler_file_path = os.path.join(
    "mlflow/pytorch/torchserve/tests/resources", "linear_handler.py"
)
sample_input_file = os.path.join(
    "mlflow/pytorch/torchserve/tests/resources", "sample.json"
)


@pytest.fixture(scope="session")
def start_torchserve():
    if not os.path.isdir("model_store"):
        os.makedirs("model_store")
    cmd = "torchserve --start --model-store {}".format("./model_store")
    _ = subprocess.Popen(cmd, shell=True).wait()

    count = 0
    for _ in range(5):
        value = health_checkup()
        if (
            value is not None
            and value != ""
            and json.loads(value)["status"] == "Healthy"
        ):
            time.sleep(1)
            break
        else:
            count += 1
            time.sleep(5)
    if count >= 5:
        raise Exception("Unable to connect to torchserve")
    return True


def health_checkup():
    curl_cmd = "curl http://localhost:8080/ping"
    (value, _) = subprocess.Popen(
        [curl_cmd], stdout=subprocess.PIPE, shell=True
    ).communicate()
    return value.decode("utf-8")


def stop_torchserve():
    cmd = "torchserve --stop"
    _ = subprocess.Popen(cmd, shell=True).wait()

    if os.path.isdir("model_store"):
        shutil.rmtree("model_store")


atexit.register(stop_torchserve)


@pytest.mark.usefixtures("start_torchserve")
def test_create_cli_success():
    version = "VERSION={model_version}".format(model_version=model_version)
    model_file = "MODEL_FILE={model_file_path}".format(model_file_path=model_file_path)
    handler_file = "HANDLER_FILE={handler_file_path}".format(handler_file_path=handler_file_path)
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
            version,
            "-C",
            model_file,
            "-C",
            handler_file,
        ],
    )
    assert "{} deployment {} is created".format(f_flavor, f_deployment_id) in res.stdout


def test_create_cli_no_version_success():
    model_file = "MODEL_FILE={model_file_path}".format(model_file_path=model_file_path)
    handler_file = "HANDLER_FILE={handler_file_path}".format(handler_file_path=handler_file_path)
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
    assert "{} deployment {} is created".format(f_flavor, f_deployment_id) in res.stdout


def test_update_cli_success():
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
            f_deployment_id,
        ],
    )
    assert (
        "Deployment {} is updated (with flavor {})".format(f_deployment_id, f_flavor)
        in res.stdout
    )


def test_list_cli_success():
    runner = CliRunner()
    res = runner.invoke(cli.list_deployment, ["--target", f_target])
    assert "{}".format(f_deployment_id) in res.stdout


def test_get_cli_success():
    runner = CliRunner()
    res = runner.invoke(
        cli.get_deployment, ["--name", f_deployment_id, "--target", f_target]
    )
    assert "{}".format(f_deployment_id) in res.stdout


def test_delete_cli_success():
    version = "VERSION=2.0"
    runner = CliRunner()
    res = runner.invoke(
        cli.delete_deployment,
        ["--name", f_deployment_id, "--target", f_target, "-C", version],
    )
    assert "Deployment {} is deleted".format(f_deployment_id) in res.stdout


def test_delete_no_version_cli_success():
    runner = CliRunner()
    res = runner.invoke(
        cli.delete_deployment,
        ["--name", f_deployment_id, "--target", f_target],
    )
    assert "Deployment {} is deleted".format(f_deployment_id) in res.stdout
