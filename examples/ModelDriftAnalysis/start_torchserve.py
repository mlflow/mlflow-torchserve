import atexit
import os
import shutil
import subprocess
import json
import time


def start_torchserve():
    if not os.path.isdir("model_store"):
        os.makedirs("model_store")
        print("model_store created")
    cmd = "torchserve --start --model-store {}".format("./model_store")
    return_code = subprocess.Popen(cmd, shell=True).wait()
    print("TorchServe Server Started")

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
    curl_cmd = "curl http://127.0.0.1:8080/ping"
    (value, err) = subprocess.Popen([curl_cmd], stdout=subprocess.PIPE, shell=True).communicate()
    return value.decode("utf-8")


def stop_torchserve():
    cmd = "torchserve --stop"
    return_code = subprocess.Popen(cmd, shell=True).wait()


#    if os.path.isdir("model_store"):
#        shutil.rmtree("model_store")


atexit.register(stop_torchserve)
