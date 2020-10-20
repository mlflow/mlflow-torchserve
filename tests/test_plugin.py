import atexit
import json
import os
import shutil
import subprocess
import time
import torch

import pytest

from mlflow import deployments
from mlflow.exceptions import MlflowException

f_target = "torchserve"
f_deployment_id = "test"
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
sample_output_file = os.path.join(
    "mlflow/pytorch/torchserve/tests/resources", "output.json"
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
def test_create_deployment_success():
    client = deployments.get_deploy_client(f_target)
    ret = client.create_deployment(
        f_deployment_id,
        f_model_uri,
        f_flavor,
        config={
            "VERSION": model_version,
            "MODEL_FILE": model_file_path,
            "HANDLER_FILE": handler_file_path,
        },
    )
    assert isinstance(ret, dict)
    assert ret["name"] == f_deployment_id
    assert ret["flavor"] == f_flavor


def test_create_deployment_no_version():
    client = deployments.get_deploy_client(f_target)
    ret = client.create_deployment(
        f_deployment_id,
        f_model_uri,
        f_flavor,
        config={
            "MODEL_FILE": model_file_path,
            "HANDLER_FILE": handler_file_path,
        },
    )
    assert isinstance(ret, dict)
    assert ret["name"] == f_deployment_id
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


def test_get_success():
    client = deployments.get_deploy_client(f_target)
    ret = client.get_deployment(f_deployment_id)
    print("Return value is ", json.loads(ret["deploy"]))
    assert json.loads(ret["deploy"])[0]["modelName"] == f_deployment_id


def test_wrong_target_name():
    with pytest.raises(MlflowException):
        deployments.get_deploy_client("wrong_target")


def test_update_deployment_success():
    client = deployments.get_deploy_client(f_target)
    ret = client.update_deployment(f_deployment_id)
    assert ret["flavor"] is None


def test_predict_success():
    client = deployments.get_deploy_client(f_target)
    with open(sample_input_file) as fp:
        data = fp.read()
    pred = client.predict(f_deployment_id, data)
    assert pred is not None


def test_predict_tensor_input():
    client = deployments.get_deploy_client(f_target)
    data = torch.Tensor([5000])
    pred = client.predict(f_deployment_id, data)
    assert pred is not None


def test_delete_success():
    client = deployments.get_deploy_client(f_target)
    assert client.delete_deployment(f_deployment_id, config={"VERSION" : "2.0"}) is None


def test_delete_no_version():
    client = deployments.get_deploy_client(f_target)
    assert client.delete_deployment(f_deployment_id) is None


f_dummy = "dummy"


def test_create_no_handler_exception():
    with pytest.raises(Exception, match="Config Variable HANDLER_FILE - missing"):
        client = deployments.get_deploy_client(f_target)
        client.create_deployment(
            f_deployment_id,
            f_model_uri,
            f_flavor,
            config={"VERSION": model_version, "MODEL_FILE": model_file_path},
        )


def test_create_no_model_exception():
    with pytest.raises(Exception, match="Config Variable MODEL_FILE - missing"):
        client = deployments.get_deploy_client(f_target)
        client.create_deployment(
            f_deployment_id,
            f_model_uri,
            f_flavor,
            config={"VERSION": model_version, "HANDLER_FILE": handler_file_path},
        )


def test_create_wrong_handler_exception():
    with pytest.raises(Exception, match="Unable to create mar file"):
        client = deployments.get_deploy_client(f_target)
        client.create_deployment(
            f_deployment_id,
            f_model_uri,
            f_flavor,
            config={
                "VERSION": model_version,
                "MODEL_FILE": model_file_path,
                "HANDLER_FILE": f_dummy,
            },
        )


def test_create_wrong_model_exception():
    with pytest.raises(Exception, match="Unable to create mar file"):
        client = deployments.get_deploy_client(f_target)
        client.create_deployment(
            f_deployment_id,
            f_model_uri,
            f_flavor,
            config={
                "VERSION": model_version,
                "MODEL_FILE": f_dummy,
                "HANDLER_FILE": handler_file_path,
            },
        )


def test_create_mar_file_exception():
    with pytest.raises(Exception, match="No such file or directory"):
        client = deployments.get_deploy_client(f_target)
        client.create_deployment(
            f_deployment_id,
            f_dummy,
            config={
                "VERSION": model_version,
                "MODEL_FILE": model_file_path,
                "HANDLER_FILE": handler_file_path,
            },
        )


def test_update_invalid_name():
    with pytest.raises(
        Exception, match="Unable to update deployment with name %s" % f_dummy
    ):
        client = deployments.get_deploy_client(f_target)
        client.update_deployment(f_dummy)


def test_get_invalid_name():
    with pytest.raises(
        Exception, match="Unable to get deployments with name %s" % f_dummy
    ):
        client = deployments.get_deploy_client(f_target)
        client.get_deployment(f_dummy)


def test_delete_invalid_name():
    with pytest.raises(
        Exception, match="Unable to delete deployment for name %s" % f_dummy
    ):
        client = deployments.get_deploy_client(f_target)
        client.delete_deployment(f_dummy, config={"VERSION": model_version})


def test_predict_exception():
    with pytest.raises(
        Exception, match="Unable to parse input json string"
    ):
        client = deployments.get_deploy_client(f_target)
        client.predict(f_dummy, "sample")


def test_predict_name_exception():
    with pytest.raises(
        Exception, match="Unable to infer the results for the name %s" % f_dummy
    ):
        client = deployments.get_deploy_client(f_target)
        with open(sample_input_file) as fp:
            data = fp.read()
        client.predict(f_dummy, data, config={"VERSION": model_version})
