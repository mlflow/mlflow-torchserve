import atexit
import json
import os
import shutil
import subprocess
import time

import pytest

from tests.resources import linear_model


@pytest.fixture(scope="session")
def start_torchserve():
    linear_model.main()
    if not os.path.isdir("model_store"):
        os.makedirs("model_store")
    cmd = "torchserve --ncs --start --model-store {}".format("./model_store")
    _ = subprocess.Popen(cmd, shell=True).wait()

    count = 0
    for _ in range(5):
        value = health_checkup()
        if value is not None and value != "" and json.loads(value)["status"] == "Healthy":
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
    (value, _) = subprocess.Popen([curl_cmd], stdout=subprocess.PIPE, shell=True).communicate()
    return value.decode("utf-8")


def stop_torchserve():
    cmd = "torchserve --stop"
    _ = subprocess.Popen(cmd, shell=True).wait()

    if os.path.isdir("model_store"):
        shutil.rmtree("model_store")

    if os.path.exists("tests/resources/linear_state_dict.pt"):
        os.remove("tests/resources/linear_state_dict.pt")

    if os.path.exists("tests/resources/linear_model.pt"):
        os.remove("tests/resources/linear_model.pt")


atexit.register(stop_torchserve)
