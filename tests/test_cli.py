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
incorrect_model_file_path = os.path.join("tests/resources", "linear_model1.py")
handler_file_path = os.path.join("tests/resources", "linear_handler.py")
sample_input_file = os.path.join("tests/resources", "sample.json")
sample_incorrect_input_file = os.path.join("tests/resources", "sample.txt")
handler_file = "HANDLER={handler_file_path}".format(handler_file_path=handler_file_path)
model_file = "MODEL_FILE={model_file_path}".format(model_file_path=model_file_path)
incorrect_model_file = "MODEL_FILE={model_file_path}".format(
    model_file_path=incorrect_model_file_path
)
runner = CliRunner()


@pytest.mark.usefixtures("start_torchserve")
def test_create_cli_version_success():
    version = "VERSION={version}".format(version="1.0")
    _ = deployments.get_deploy_client(f_target)
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
    _ = deployments.get_deploy_client(f_target)
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


def test_create_cli_failure_without_version():
    _ = deployments.get_deploy_client(f_target)
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
            incorrect_model_file,
            "-C",
            handler_file,
        ],
    )
    assert "No such file or directory" in str(res.exception) and res.exit_code == 1
    res = runner.invoke(
        cli.create_deployment,
        [
            "-m",
            f_model_uri,
            "-t",
            f_target,
            "--name",
            f_deployment_id,
            "-C",
            model_file,
        ],
    )
    assert str(res.exception) == "Config Variable HANDLER - missing"
    res = runner.invoke(cli.create_deployment)
    assert (
        res.exit_code == 2
        and res.output == "Usage: create [OPTIONS]\nTry 'create --help' for help.\n\n"
        "Error: Missing option '--name'.\n"
    )
    res = runner.invoke(
        cli.create_deployment,
        [
            "-m",
            f_model_uri,
            "-t",
            f_target,
            "--name",
            f_deployment_id,
            "-C",
            handler_file,
        ],
    )
    assert str(res.exception) == "Unable to register the model"
    res = runner.invoke(
        cli.create_deployment,
        [
            "-t",
            f_target,
            "--name",
            f_deployment_id,
            "-C",
            handler_file,
            "-C",
            model_file,
        ],
    )
    assert res.exit_code == 2
    res = runner.invoke(
        cli.create_deployment,
        [
            "-m",
            f_model_uri,
            "--name",
            f_deployment_id,
            "-C",
            handler_file,
            "-C",
            model_file,
        ],
    )
    assert res.exit_code == 2
    res = runner.invoke(
        cli.create_deployment,
        [
            "-m",
            f_model_uri,
            "-t",
            f_target,
            "-C",
            handler_file,
            "-C",
            handler_file,
        ],
    )
    assert res.exit_code == 2


@pytest.mark.parametrize(
    "deployment_name, config",
    [(f_deployment_name_version, "SET-DEFAULT=true"), (f_deployment_id, "MIN_WORKER=3")],
)
def test_update_cli_success(deployment_name, config):
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
    res = runner.invoke(cli.list_deployment, ["--target", f_target])
    assert "{}".format(f_deployment_id) in res.stdout


@pytest.mark.parametrize(
    "deployment_name", [f_deployment_id, f_deployment_name_version, f_deployment_name_all]
)
def test_get_cli_success(deployment_name):
    res = runner.invoke(cli.get_deployment, ["--name", deployment_name, "--target", f_target])
    ret = json.loads(res.stdout.split("deploy:")[1])
    if deployment_name == f_deployment_id:
        assert ret[0]["modelName"] == f_deployment_id
    elif deployment_name == f_deployment_name_version:
        assert ret[0]["modelVersion"] == f_deployment_name_version.split("/")[1]
    else:
        assert len(ret) == 2


@pytest.mark.parametrize(
    "deployment_name", [f_deployment_id, f_deployment_name_version, f_deployment_name_all]
)
def test_get_cli_failure(deployment_name):
    res = runner.invoke(
        cli.get_deployment,
    )
    assert (
        res.exit_code == 2
        and res.output
        == "Usage: get [OPTIONS]\nTry 'get --help' for help.\n\nError: Missing option '--name'.\n"
    )
    res = runner.invoke(cli.get_deployment, ["--name", deployment_name])
    assert (
        res.exit_code == 2
        and res.output == "Usage: get [OPTIONS]\nTry 'get --help' for help.\n\n"
        "Error: Missing option '--target' / '-t'.\n"
    )


@pytest.mark.parametrize("deployment_name", [f_deployment_name_version, f_deployment_id])
def test_predict_cli_success(deployment_name):
    res = runner.invoke(
        cli.predict,
        ["--name", deployment_name, "--target", f_target, "--input-path", sample_input_file],
    )
    assert res.exit_code == 0


@pytest.mark.parametrize("deployment_name", [f_deployment_name_version, f_deployment_id])
def test_predict_cli_failure(deployment_name):
    res = runner.invoke(
        cli.predict,
        ["--name", deployment_name, "--target", f_target],
    )
    assert (
        res.exit_code == 2
        and res.output == "Usage: predict [OPTIONS]\nTry 'predict --help' for help.\n\n"
        "Error: Missing option '--input-path' / '-I'.\n"
    )
    res = runner.invoke(
        cli.predict,
        ["--name", deployment_name, "--input-path", sample_input_file],
    )
    assert (
        res.exit_code == 2
        and res.output == "Usage: predict [OPTIONS]\nTry 'predict --help' for help.\n\n"
        "Error: Missing option '--target' / '-t'.\n"
    )
    res = runner.invoke(
        cli.predict,
        ["--target", f_target, "--input-path", sample_input_file],
    )
    assert (
        res.exit_code == 2
        and res.output == "Usage: predict [OPTIONS]\nTry 'predict --help' for help.\n\n"
        "Error: Missing option '--name'.\n"
    )
    res = runner.invoke(
        cli.predict,
        [
            "--name",
            deployment_name,
            "--target",
            f_target,
            "--input-path",
            sample_incorrect_input_file,
        ],
    )
    assert res.exception


@pytest.mark.parametrize("deployment_name", [f_deployment_name_version, f_deployment_id])
def test_explain_cli_success(deployment_name):
    runner.invoke(
        cli.explain,
        ["--name", deployment_name, "--target", f_target, "--input-path", sample_input_file],
    )


@pytest.mark.parametrize("deployment_name", [f_deployment_name_version, f_deployment_id])
def test_explain_cli_failure(deployment_name):
    res = runner.invoke(
        cli.explain,
        ["--name", deployment_name, "--target", f_target],
    )
    assert (
        res.exit_code == 2
        and res.output == "Usage: explain [OPTIONS]\nTry 'explain --help' for help.\n\n"
        "Error: Missing option '--input-path' / '-I'.\n"
    )
    res = runner.invoke(
        cli.explain,
        ["--name", deployment_name, "--input-path", sample_input_file],
    )
    assert (
        res.exit_code == 2
        and res.output == "Usage: explain [OPTIONS]\nTry 'explain --help' for help.\n\n"
        "Error: Missing option '--target' / '-t'.\n"
    )
    res = runner.invoke(
        cli.explain,
        [
            "--name",
            deployment_name,
            "--target",
            f_target,
            "--input-path",
            sample_incorrect_input_file,
        ],
    )
    assert res.exception


@pytest.mark.parametrize("deployment_name", [f_deployment_id + "/1.0", f_deployment_name_version])
def test_delete_cli_success(deployment_name):
    res = runner.invoke(
        cli.delete_deployment,
        ["--name", deployment_name, "--target", f_target],
    )
    assert "Deployment {} is deleted".format(deployment_name) in res.stdout


@pytest.mark.parametrize("deployment_name", [f_deployment_id + "/1.0", f_deployment_name_version])
def test_delete_cli_failure(deployment_name):
    res = runner.invoke(
        cli.delete_deployment,
        ["--name", deployment_name],
    )
    assert (
        res.exit_code == 2
        and res.output == "Usage: delete [OPTIONS]\nTry 'delete --help' for help.\n\n"
        "Error: Missing option '--target' / '-t'.\n"
    )

    res = runner.invoke(
        cli.delete_deployment,
        ["--target", f_target],
    )
    assert (
        res.exit_code == 2
        and res.output == "Usage: delete [OPTIONS]\nTry 'delete --help' for help.\n\n"
        "Error: Missing option '--name'.\n"
    )

    res = runner.invoke(
        cli.delete_deployment,
    )
    assert (
        res.exit_code == 2
        and res.output == "Usage: delete [OPTIONS]\nTry 'delete --help' for help.\n\n"
        "Error: Missing option '--name'.\n"
    )
