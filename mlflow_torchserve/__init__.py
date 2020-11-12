import docker
import json
import subprocess
import time
import logging
import os
import pandas as pd
import torch
from pathlib import Path

import requests
from mlflow_torchserve.config import Config

from mlflow.deployments import BaseDeploymentClient, get_deploy_client
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.models.model import Model

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

        if "handler" not in self.server_config:
            raise Exception("Config Variable HANDLER - missing")

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

        if "requirements_file" not in self.server_config:
            self.server_config["requirements_file"] = None

        if "model_file" not in self.server_config:
            self.server_config["model_file"] = None

        mar_file_path = self.__generate_mar_file(
            model_name=name,
            version=str(version),
            model_file=self.server_config["model_file"],
            handler=self.server_config["handler"],
            requirements_file=self.server_config["requirements_file"],
            extra_files=self.server_config["extra_files"],
            model_uri=model_uri,
        )

        config_registration = {
            key: val
            for key, val in config.items()
            if key.upper()
            not in ["VERSION", "MODEL_FILE", "HANDLER", "EXTRA_FILES", "REQUIREMENTS_FILE"]
        }

        self.__register_model(
            mar_file_path=mar_file_path,
            config=config_registration,
        )

        return {"name": name + "/" + str(version), "flavor": flavor}

    # pylint: disable=W0221
    def delete_deployment(self, name):
        """
        Delete the deployment with the name given at --name from the specified target

        :param name: Name of the of the model with version number. For ex: "mnist/2.0"

        :return: None
        """
        url = "{api}/{models}/{name}".format(api=self.management_api, models="models", name=name)
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

        :param name: Name and version number of the model. Version number is optional. \n
                     Ex: "mnist" - Default version is updated based on config params \n
                     "mnist/2.0" - mnist 2.0 version is updated based on config params \n
        :param model_uri: Model uri cannot be updated and torchserve plugin doesnt use
                          this argument. Added to match base signature
        :param flavor: torchserve plugin doesnt use
                          this argument. Added to match base signature
        :param config: Configuration parameters like model file path, handler path

        :return: output - Returns a dict with flavor as key
        """

        query_path = ""

        if config is not None:
            for key in config:
                if key.lower() != "set-default":
                    query_path += "&" + key + "=" + str(config[key])

            if query_path:
                query_path = query_path[1:]

        url = "{api}/{models}/{name}".format(api=self.management_api, models="models", name=name)

        if config and "set-default" in [key.lower() for key in config.keys()]:
            url = "{url}/set-default".format(url=url)

        if query_path:
            url = "{url}?{query_path}".format(url=url, query_path=query_path)

        resp = requests.put(url)

        if resp.status_code not in [200, 202]:
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

        :param name: Name and version of the model. \n
                     Ex: "mnist/3.0" - gets the details of mnist model version 3.0 \n
                     "mnist" - gets the details of the default version of the model \n
                     "mnist/all" - gets the details of all the versions of the same model \n

        :return: output - Returns a dict with deploy as key and info
                          about the model specified as value
        """

        url = "{api}/{models}/{name}".format(api=self.management_api, models="models", name=name)
        resp = requests.get(url)
        if resp.status_code != 200:
            raise ValueError(
                "Unable to get deployments with name %s. "
                "Server returned status code %s and response: %s"
                % (name, resp.status_code, resp.content)
            )
        return {"deploy": resp.text}

    def predict(self, deployment_name, df):
        """
        Predict using the inference api
        Takes dataframe, Tensor or json string as input and returns output as string

        :param deployment_name: Name and version number of the deployment \n
                                Ex: "mnist/2.0" - predict based on mnist version 2.0 \n
                                "mnist" - predict based on default version. \n
        :param df: Dataframe object or json object as input

        :return: output - Returns the predicted value
        """

        url = "{api}/{predictions}/{name}".format(
            api=self.inference_api, predictions="predictions", name=deployment_name
        )
        if isinstance(df, pd.DataFrame):
            df = df.to_json(orient="records")[1:-1].replace("},{", "} {")

        if torch.is_tensor(df):
            data = json.dumps({"data": df.tolist()})
        else:
            try:
                data = json.loads(df)
            except TypeError as e:
                raise TypeError("Input data can either be dataframe or Json string: {}".format(e))
            except json.decoder.JSONDecodeError as e:
                raise ValueError("Unable to parse input json string: {}".format(e))

        resp = requests.post(url, data)
        if resp.status_code != 200:
            raise Exception(
                "Unable to infer the results for the name %s. "
                "Server returned status code %s and response: %s"
                % (deployment_name, resp.status_code, resp.content)
            )

        return resp.text

    def __generate_mar_file(
        self, model_name, version, model_file, handler, requirements_file, extra_files, model_uri
    ):

        """
        Generates mar file using the torch archiver in the specified model store path
        """
        valid_file_suffixes = [".pt", ".pth"]
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

                    model = Model.load(model_config)
                    model_json = json.loads(Model.to_json(model))

                    try:
                        if model_json["flavors"]["pytorch"]["extra_files"]:
                            for extra_file in model_json["flavors"]["pytorch"]["extra_files"]:
                                extra_files_list.append(os.path.join(path, extra_file["path"]))
                    except KeyError:
                        pass

                    try:
                        if model_json["flavors"]["pytorch"]["requirements_file"]:
                            req_file_path = os.path.join(
                                path, model_json["flavors"]["pytorch"]["requirements_file"]["path"]
                            )
                    except KeyError:
                        pass

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
            "--version {} --serialized-file {} "
            "--handler {} --export-path {}".format(
                model_name, version, model_uri, handler, model_store
            )
        )

        if model_file:
            cmd += " --model-file {file_path}".format(file_path=model_file)

        extra_files_str = ""
        if extra_files_list:
            extra_files_str += ",".join(extra_files_list)

        if extra_files:
            if extra_files_list:
                extra_files_str = "{base_string},{user_defined_string}".format(
                    base_string=extra_files_str,
                    user_defined_string=str(extra_files).replace("'", ""),
                )
            else:
                extra_files_str = str(extra_files).replace("'", "")

        if extra_files_str:
            cmd = "{cmd} --extra-files '{extra_files}'".format(cmd=cmd, extra_files=extra_files_str)

        if requirements_file:
            cmd = "{cmd} -r {path}".format(cmd=cmd, path=requirements_file)
        elif req_file_path:
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
            get_response = self.get_deployment(name + "/all")
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
        url = "http://localhost:{port}/ping".format(port=_DEFAULT_TORCHSERVE_LOCAL_INFERENCE_PORT)
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
        "README https://github.com/mlflow/mlflow-torchserve/blob/master/README.md \n\n"
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
