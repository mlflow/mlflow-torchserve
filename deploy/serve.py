import docker
import json
import subprocess
import time
import logging
import os
import pandas as pd
import torch
from pathlib import Path, PurePath

import requests
from deploy.config import Config

from mlflow.deployments import BaseDeploymentClient, get_deploy_client
from mlflow.tracking.artifact_utils import _download_artifact_from_uri

_logger = logging.getLogger(__name__)

_DEFAULT_TORCHSERVE_LOCAL_INFERENCE_PORT = "8080"
_DEFAULT_TORCHSERVE_LOCAL_MANAGEMENT_PORT = "8081"


class TorchServePlugin(BaseDeploymentClient):
    def __init__(self, uri):

        """
        Initializes the deployment plugin and sets the environment variables
        """
        super(TorchServePlugin, self).__init__(target_uri=uri)
        self.server_config = Config()
        self.inference_api, self.management_api = self.__get_torch_serve_port()
        self.default_limit = 100

    def __get_torch_serve_port(self):
        """
        Reads through the config properties for torchserve inference and management api's
        """
        config_properties = self.server_config["config_properties"]
        inference_port = "http://localhost:8080"
        management_port = "http://localhost:8081"
        address_strings = self.server_config["torchserve_address_names"]
        if config_properties is not None and os.path.exists(config_properties):
            with open(config_properties, "r") as f:
                lines = f.readlines()
                for line in lines:
                    name = line.strip().split("=")
                    if name[0] == address_strings[0] and name[1] is not None:
                        inference_port = name[1]
                    if name[0] == address_strings[1] and name[1] is not None:
                        management_port = name[1]
        return inference_port, management_port

    def __validate_mandatory_arguments(self):
        """
        Validate the mandatory arguments is present if not raise exception
        """

        if "model_file" not in self.server_config:
            raise Exception("Config Variable MODEL_FILE - missing")

        if "handler_file" not in self.server_config:
            raise Exception("Config Variable HANDLER_FILE - missing")

    def create_deployment(self, name, model_uri, flavor=None, config=None):
        """
        Deploy the model at the model_uri to the specified target

        :param name: Name of the of the model
        :param model_uri: Serialized python file [.pt or .pth]
        :param flavor: Flavor of the deployed model
        :param config: Configuration parameters like model file path, handler path

        :return: output - Returns a dict with flavor and name as keys
        """

        version = 1.0
        is_version_provided = False
        if config:
            for key in config:
                if key.upper() == "VERSION":
                    version = float(config[key])
                    is_version_provided = True
                self.server_config[key.lower()] = str(config[key])

        self.__validate_mandatory_arguments()

        if not is_version_provided:
            version = self.__get_max_version(name)

        if "extra_files" not in self.server_config:
            self.server_config["extra_files"] = None

        if "requirements" not in self.server_config:
            self.server_config["requirements"] = None

        mar_file_path = self.__generate_mar_file(
            model_name=name,
            version=str(version),
            model_file=self.server_config["model_file"],
            handler_file=self.server_config["handler_file"],
            requirements=self.server_config["requirements"],
            extra_files=self.server_config["extra_files"],
            model_uri=model_uri
        )

        config_registration = {
            key: val
            for key, val in config.items()
            if key.upper() not in ["VERSION", "MODEL_FILE", "HANDLER_FILE", "EXTRA_FILES", "REQUIREMENTS"]
        }

        self.__register_model(
            mar_file_path=mar_file_path,
            config=config_registration,
        )

        return {"name": name, "flavor": flavor}

    # pylint: disable=W0221
    def delete_deployment(self, name, config=None):
        """
        Delete the deployment with the name given at --name from the specified target

        :param name: Name of the of the model
        :param config: Configuration parameters like model file path, handler path

        :return: None
        """

        version = "1.0"
        if config:
            for key in config:
                if key.upper() == "VERSION":
                    version = str(config[key])

        url = "{}/{}/{}/{}".format(self.management_api, "models", name, version)
        resp = requests.delete(url)
        if resp.status_code != 200:
            raise Exception(
                "Unable to delete deployment for name %s. "
                "Server returned status code %s and response: %s"
                % (name, resp.status_code, resp.content)
            )
        return None

    def update_deployment(self, name, model_uri=None, flavor=None, config=None):
        """
        Update the deployment with the name given at --name from the specified target
        Using -C or --config additional parameters shall be updated for the corresponding model

        :param name: Name of the of the model
        :param model_uri: Serialized python file [.pt or .pth]
        :param flavor: Flavor of the deployed model
        :param config: Configuration parameters like model file path, handler path

        :return: output - Returns a dict with flavor as key
        """

        query_path = ""

        if config is not None:
            for key in config:
                query_path += "&" + key + "=" + str(config[key])

            query_path = query_path[1:]

        url = "{}/{}/{}?{}".format(self.management_api, "models", name, query_path)
        resp = requests.put(url)

        if resp.status_code != 202:
            raise Exception(
                "Unable to update deployment with name %s. "
                "Server returned status code %s and response: %s"
                % (name, resp.status_code, resp.content)
            )
        return {"flavor": flavor}

    def list_deployments(self):
        """
        List the names of all model deployments in the specified target.
        These names can be used with delete, update and get commands

        :return: output - Returns a list of models of the target
        """

        deployment_list = []
        limit = self.default_limit
        nextPageToken = 0

        while True:
            url = "{}/{}".format(self.management_api, "models")

            input_params = {"limit": limit, "next_page_token": nextPageToken}

            resp = requests.get(url, params=input_params)
            if resp.status_code != 200:
                raise Exception("Unable to list deployments")
            temp = json.loads(resp.text)
            model_count = len(temp["models"])
            for i in range(model_count):
                tempDict = {}
                key = temp["models"][i]["modelName"]
                tempDict[key] = temp["models"][i]
                deployment_list.append(tempDict)
            if "nextPageToken" not in temp:
                break
            else:
                nextPageToken = temp["nextPageToken"]
        return deployment_list

    def get_deployment(self, name):
        """
        Print the detailed description of the deployment with the name given at --name
        in the specified target

        :param name: Name of the of the model

        :return: output - Returns a dict with deploy as key and info about the model specified as value
        """

        url = "{}/{}/{}/{}".format(self.management_api, "models", name, "all")
        resp = requests.get(url)
        if resp.status_code != 200:
            raise ValueError(
                "Unable to get deployments with name %s. "
                "Server returned status code %s and response: %s"
                % (name, resp.status_code, resp.content)
            )
        return {"deploy": resp.text}

    def predict(self, deployment_name, df, config=None):
        """
        Predict using the inference api
        Takes dataframe, Tensor or json string as input and returns output as string

        :param deployment_name: Name of the of the model
        :param df: Dataframe object or json object as input
        :param config: Configuration parameters like model version

        :return: output - Returns the predicted value
        """

        version = "1.0"
        if config:
            for key in config:
                if key.upper() == "VERSION":
                    version = str(config[key])

        url = "{}/{}/{}/{}".format(
            self.inference_api, "predictions", deployment_name, version
        )
        if isinstance(df, pd.DataFrame):
            df = df.to_json(orient="records")[1:-1].replace("},{", "} {")

        if torch.is_tensor(df):
            data = json.dumps({"data": df.tolist()})
        else:
            try:
                data = json.loads(df)
            except TypeError as e:
                raise TypeError(
                    "Input data can either be dataframe or Json string: {}".format(e)
                )
            except json.decoder.JSONDecodeError as e:
                raise ValueError(
                    "Unable to parse input json string: {}".format(e)
                )

        resp = requests.post(url, data)
        if resp.status_code != 200:
            raise Exception(
                "Unable to infer the results for the name %s. "
                "Server returned status code %s and response: %s"
                % (deployment_name, resp.status_code, resp.content)
            )

        return resp.text

    def __generate_mar_file(
        self, model_name, version, model_file, handler_file, requirements, extra_files, model_uri
    ):

        """
        Generates mar file using the torch archiver in the specified model store path
        """
        valid_file_suffixes = [".pt", ".pth"]
        requirements_file = "requirements.txt"
        requirements_directory_name = "requirements"
        extra_files_directory_name = "artifacts"
        extra_files_list = []
        req_file_path = None

        if not os.path.isfile(model_uri):
            path = Path(_download_artifact_from_uri(model_uri))
            model_config = path / "MLmodel"
            if not model_config.exists():
                raise Exception(
                    "Failed to find MLmodel configuration within "
                    "the specified model's root directory."
                )
            else:
                model_path = None
                if path.suffix in valid_file_suffixes:
                    model_uri = path
                else:
                    for root, dirs, files in os.walk(path, topdown=False):
                        for name in files:
                            if Path(name).suffix in valid_file_suffixes:
                                model_path = os.path.join(root, name)

                        for directory in dirs:
                            if directory == extra_files_directory_name:
                                dir_list = os.path.join(root, directory)
                                for extra_file in os.listdir(dir_list):
                                    extra_files_list.append(os.path.join(dir_list, extra_file))
                                    if extra_files is None:
                                        extra_files = True

                            if directory == requirements_directory_name:
                                dir_list = os.path.join(root, directory)
                                for requirements_file in os.listdir(dir_list):
                                    req_file_path = os.path.join(dir_list, requirements_file)
                                    if requirements is None:
                                        requirements = True
                    if model_path is None:
                        raise RuntimeError(
                            "Model file does not have a valid suffix. Expected to be one of "
                            + ", ".join(valid_file_suffixes)
                        )
                    model_uri = model_path

        export_path = self.server_config["export_path"]
        if export_path:
            model_store = export_path
        else:
            model_store = "model_store"
            if not os.path.isdir(model_store):
                os.makedirs(model_store)

        cmd = (
            "torch-model-archiver --force --model-name {} "
            "--version {} --model-file {} --serialized-file {} "
            "--handler {} --export-path {}".format(
                model_name, version, model_file, model_uri, handler_file, model_store
            )
        )
        if extra_files:
            extra_files_str = ""
            if type(extra_files) == str:
                extra_files_str += str(extra_files).replace('\'', "")
                if len(extra_files_list) > 0:
                    extra_files_str += ","
            if len(extra_files_list) > 0:
                extra_files_str += ",".join(extra_files_list)
            cmd = "{cmd} --extra-files '{extra_files}'".format(cmd=cmd, extra_files=extra_files_str)

        if req_file_path:
            cmd = "{cmd} -r {path}".format(cmd=cmd, path=req_file_path)

        return_code = subprocess.Popen(cmd, shell=True).wait()
        if return_code != 0:
            _logger.error(
                "Error when attempting to load and parse JSON cluster spec from file %s",
                cmd,
            )
            raise Exception("Unable to create mar file")

        if export_path:
            mar_file = "{}/{}.mar".format(export_path, model_name)
        else:
            mar_file = "{}/{}/{}.mar".format(os.getcwd(), model_store, model_name)
            if os.path.isfile(mar_file):
                print("{} file generated successfully".format(mar_file))

        return mar_file

    def __register_model(self, mar_file_path, config=None):
        """
        Register the model using the mar file that has been generated by the archiver
        """
        query_path = mar_file_path
        if config:
            for key in config:
                query_path += "&" + key + "=" + str(config[key])
        else:
            query_path += "&initial_workers=" + str(1)

        url = "{}/{}?url={}".format(self.management_api, "models", query_path)

        resp = requests.post(url=url)

        if resp.status_code != 200:
            raise Exception("Unable to register the model")
        return True

    def __get_max_version(self, name):
        """
        Returns the maximum version for the same model name
        """
        try:
            get_response = self.get_deployment(name)
        except ValueError:
            get_response = None

        max_version = 1.0

        if get_response:
            for model in json.loads(get_response["deploy"]):
                model_version = float(model["modelVersion"])
                if model_version > max_version:
                    max_version = model_version
            max_version += 1.0
        return max_version


def run_local(name, model_uri, flavor=None, config=None):
    device = config.get("device", "cpu")
    if device.lower() == "gpu":
        docker_image = "pytorch/torchserve:latest-gpu"
    else:
        docker_image = "pytorch/torchserve:latest"

    client = docker.from_env()
    client.containers.run(
        image=docker_image,
        auto_remove=True,
        ports={
            _DEFAULT_TORCHSERVE_LOCAL_INFERENCE_PORT: _DEFAULT_TORCHSERVE_LOCAL_INFERENCE_PORT,
            _DEFAULT_TORCHSERVE_LOCAL_MANAGEMENT_PORT: _DEFAULT_TORCHSERVE_LOCAL_MANAGEMENT_PORT,
        },
        detach=True,
    )

    for _ in range(10):
        url = "http://localhost:{port}/ping".format(
            port=_DEFAULT_TORCHSERVE_LOCAL_INFERENCE_PORT
        )
        try:
            resp = requests.get(url)
            if resp.status_code != 200:
                time.sleep(6)
                continue
            else:
                break
        except requests.exceptions.ConnectionError:
            time.sleep(6)
    else:
        raise RuntimeError(
            "Could not start the torchserve docker container. You can "
            "try setting up torchserve locally"
            " and call the ``create`` API with target_uri as given in "
            "the example command below (this will set the host as "
            "localhost and port as 8080)\n\n"
            "    mlflow deployments create -t torchserve -m <modeluri> ...\n\n"
        )
    plugin = get_deploy_client("torchserve")
    plugin.create_deployment(name, model_uri, flavor, config)


def target_help():
    help_string = (
        "\nmlflow-torchserve plugin integrates torchserve to mlflow deployment pipeline. "
        "For detailed explanation and to see multiple examples, checkout the Readme at "
        "README https://github.com/chauhang/mlflow/blob/master/README.rst \n\n"
        "Following are the various options"
        "available using the existing "
        "mlflow deployments functions\n\n"
        "CREATE: \n"
        "Deploy the model at 'model_uri' to the "
        "specified target.\n"
        "Additional plugin-specific arguments may also be "
        "passed to this command, via -C key=value\n\n"
        "UPDATE: \n"
        "Update the deployment with ID 'deployment_id' in the specified target.\n"
        "You can update the URI of the model and/or the flavor of the deployed model"
        "(in which case the model URI must also be specified).\n"
        "Additional plugin-specific arguments may also be passed to this "
        "command, via '-C key=value'.\n\n"
        "DELETE: \n"
        "Delete the deployment with name given at '--name' from the specified target.\n\n"
        "LIST: \n"
        "List the names of all model deployments in the specified target."
        "These names can be used with the 'delete', 'update', and 'get' commands.\n\n"
        "GET: \n"
        "Print a detailed description of the deployment with name given at "
        "'--name' in the specified target.\n\n"
        "HELP: \n"
        "Display additional help for a specific deployment target, e.g. info on target-specific"
        "config options and the target's URI format.\n\n"
        "RUN-LOCAL: \n"
        "Deploy the model locally. This has very similar signature to 'create' API\n\n"
        "PREDICT: \n"
        "Predict the results for the deployed model for the given input(s)\n\n"
    )

    return help_string
