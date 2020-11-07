import json
import os

import pytest
import torch
from mlflow import deployments
from mlflow.exceptions import MlflowException

f_target = "torchserve"
f_deployment_id = "test"
f_deployment_name_version = "test/2.0"
f_deployment_name_all = "test/all"
f_flavor = None
f_model_uri = os.path.join("tests/resources", "linear_state_dict.pt")

model_version = "1.0"
model_file_path = os.path.join("tests/resources", "linear_model.py")
handler_file_path = os.path.join("tests/resources", "linear_handler.py")
sample_input_file = os.path.join("tests/resources", "sample.json")
sample_output_file = os.path.join("tests/resources", "output.json")


@pytest.mark.usefixtures("start_torchserve")
def test_create_deployment_success():
    client = deployments.get_deploy_client(f_target)
    ret = client.create_deployment(
        f_deployment_id,
        f_model_uri,
        f_flavor,
        config={
            "VERSION": model_version,
            "MODEL_FILE": model_file_path,
            "HANDLER": handler_file_path,
        },
    )
    assert isinstance(ret, dict)
    assert ret["name"] == f_deployment_id + "/" + model_version
    assert ret["flavor"] == f_flavor


def test_create_deployment_no_version():
    client = deployments.get_deploy_client(f_target)
    ret = client.create_deployment(
        f_deployment_id,
        f_model_uri,
        f_flavor,
        config={"MODEL_FILE": model_file_path, "HANDLER": handler_file_path},
    )
    assert isinstance(ret, dict)
    assert ret["name"] == f_deployment_name_version
    assert ret["flavor"] == f_flavor


def test_list_success():
    client = deployments.get_deploy_client(f_target)
    ret = client.list_deployments()
    isNamePresent = False
    for i in range(len(ret)):
        if list(ret[i].keys())[0] == f_deployment_id:
            isNamePresent = True
            break
    if isNamePresent:
        assert True
    else:
        assert False


@pytest.mark.parametrize(
    "deployment_name", [f_deployment_id, f_deployment_name_version, f_deployment_name_all]
)
def test_get_success(deployment_name):
    client = deployments.get_deploy_client(f_target)
    ret = client.get_deployment(deployment_name)
    print("Return value is ", json.loads(ret["deploy"]))
    if deployment_name == f_deployment_id:
        assert json.loads(ret["deploy"])[0]["modelName"] == f_deployment_id
    elif deployment_name == f_deployment_name_version:
        assert (
            json.loads(ret["deploy"])[0]["modelVersion"] == f_deployment_name_version.split("/")[1]
        )
    else:
        assert len(json.loads(ret["deploy"])) == 2


def test_wrong_target_name():
    with pytest.raises(MlflowException):
        deployments.get_deploy_client("wrong_target")


@pytest.mark.parametrize(
    "deployment_name, config",
    [(f_deployment_name_version, {"SET-DEFAULT": "true"}), (f_deployment_id, {"MIN_WORKER": 3})],
)
def test_update_deployment_success(deployment_name, config):
    client = deployments.get_deploy_client(f_target)
    ret = client.update_deployment(deployment_name, config)
    assert ret["flavor"] is None


@pytest.mark.parametrize("deployment_name", [f_deployment_name_version, f_deployment_id])
def test_predict_success(deployment_name):
    client = deployments.get_deploy_client(f_target)
    with open(sample_input_file) as fp:
        data = fp.read()
    pred = client.predict(deployment_name, data)
    assert pred is not None


@pytest.mark.parametrize("deployment_name", [f_deployment_name_version, f_deployment_id])
def test_predict_tensor_input(deployment_name):
    client = deployments.get_deploy_client(f_target)
    data = torch.Tensor([5000])
    pred = client.predict(deployment_name, data)
    assert pred is not None


@pytest.mark.parametrize("deployment_name", [f_deployment_name_version, f_deployment_id])
def test_delete_success(deployment_name):
    client = deployments.get_deploy_client(f_target)
    assert client.delete_deployment(deployment_name) is None


f_dummy = "dummy"


def test_create_no_handler_exception():
    client = deployments.get_deploy_client(f_target)
    with pytest.raises(Exception, match="Config Variable HANDLER - missing"):
        client.create_deployment(
            f_deployment_id,
            f_model_uri,
            f_flavor,
            config={"VERSION": model_version, "MODEL_FILE": model_file_path},
        )


def test_create_wrong_handler_exception():
    client = deployments.get_deploy_client(f_target)
    with pytest.raises(Exception, match="Unable to create mar file"):
        client.create_deployment(
            f_deployment_id,
            f_model_uri,
            f_flavor,
            config={"VERSION": model_version, "MODEL_FILE": model_file_path, "HANDLER": f_dummy},
        )


def test_create_wrong_model_exception():
    client = deployments.get_deploy_client(f_target)
    with pytest.raises(Exception, match="Unable to create mar file"):
        client.create_deployment(
            f_deployment_id,
            f_model_uri,
            f_flavor,
            config={"VERSION": model_version, "MODEL_FILE": f_dummy, "HANDLER": handler_file_path},
        )


def test_create_mar_file_exception():
    client = deployments.get_deploy_client(f_target)
    with pytest.raises(Exception, match="No such file or directory"):
        client.create_deployment(
            f_deployment_id,
            f_dummy,
            config={
                "VERSION": model_version,
                "MODEL_FILE": model_file_path,
                "HANDLER": handler_file_path,
            },
        )


def test_update_invalid_name():
    client = deployments.get_deploy_client(f_target)
    with pytest.raises(Exception, match="Unable to update deployment with name %s" % f_dummy):
        client.update_deployment(f_dummy)


def test_get_invalid_name():
    client = deployments.get_deploy_client(f_target)
    with pytest.raises(Exception, match="Unable to get deployments with name %s" % f_dummy):
        client.get_deployment(f_dummy)


def test_delete_invalid_name():
    client = deployments.get_deploy_client(f_target)
    with pytest.raises(Exception, match="Unable to delete deployment for name %s" % f_dummy):
        client.delete_deployment(f_dummy)


def test_predict_exception():
    client = deployments.get_deploy_client(f_target)
    with pytest.raises(Exception, match="Unable to parse input json string"):
        client.predict(f_dummy, "sample")


def test_predict_name_exception():
    with open(sample_input_file) as fp:
        data = fp.read()
    client = deployments.get_deploy_client(f_target)
    with pytest.raises(Exception, match="Unable to infer the results for the name %s" % f_dummy):
        client.predict(f_dummy, data)
