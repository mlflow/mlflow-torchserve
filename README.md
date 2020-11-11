# mlflow-torchserve

A plugin that integrates [TorchServe](https://github.com/pytorch/serve) with MLflow pipeline. ``mlflow_torchserve`` enables you to
use mlflow to deploy the models built and trained in mlflow pipeline into TorchServe without any
extra effort from the user. This plugin provides few command line APIs, which is also accessible
through mlflow's python package, to make the deployment process seamless.

## Installation
For installing and activating the plugin, you only need to install this package which is available
in pypi and can be installed with

```bash
pip install mlflow-torchserve
```
## What does it do
Installing this package uses python's entrypoint mechanism to register the plugin into MLflow's
plugin registry. This registry will be invoked each time you launch MLflow script or command line
argument.


### Create deployment
Deploy the model to TorchServe. The `create` command line argument and ``create_deployment`` python
APIs does the deployment of a model built with MLflow to TorchServe.

##### CLI
```shell script
mlflow deployments create -t torchserve -m <model uri> --name DEPLOYMENT_NAME -C 'MODEL_FILE=<model file path>' -C 'HANDLER=<handler file path>'
```

##### Python API
```python
from mlflow.deployments import get_deploy_client
target_uri = 'torchserve'
plugin = get_deploy_client(target_uri)
plugin.create_deployment(name=<deployment name>, model_uri=<model uri>, config={"MODEL_FILE": <model file path>, "HANDLER": <handler file path>})
```

### Update deployment
Update API can be used to update an already deployed model. This setup is useful if you want to increase the number of workers
or set a model as default version. TorchServe will make sure the user experience is seamless while changing the model in a live environment.

##### CLI
```shell script
mlflow deployments update -t torchserve --name <deployment name> -C "min-worker=<number of workers>"
```

##### Python API
```python
plugin.update_deployment(name=<deployment name>, config={'min-worker': <number of workers>})
```

### Delete deployment
Delete an existing deployment. Error will be thrown if the model is not already deployed

##### CLI
```shell script
mlflow deployments delete -t torchserve --name <deployment name / version number>
```

##### Python API
```python
plugin.delete_deployment(name=<deployment name / version number>)
```

### List all deployments
List the names of all the deploymented models. This name can then be used in other APIs or can be
used in the get deployment API to get more details about a particular deployment.

##### CLI
```shell script
mlflow deployments list -t torchserve
```

##### Python API
```python
plugin.list_deployments()
```

### Get deployment details
Get API fetches the details of the deployed model. By default, Get API fetches all the versions of the 
deployed model

##### CLI
```shell script
mlflow deployments get -t torchserve --name <deployment name>
```

##### Python API
```python
plugin.get_deployment(name=<deployment name>)
```

### Run Prediction on deployed model
Predict API enables to run prediction on the deployed model. 

CLI takes json file path as input. However, input to the python plugin can be one among the three types
DataFrame, Tensor or Json String.

##### CLI
```shell script
mlflow deployments predict -t torchserve --name <deployment name> --input-path <input file path> --output-path <output file path>
```

output-path is an optional parameter. Without output path parameter result will printed in console.

##### Python API
```python
plugin.predict(name=<deployment name>, df=<prediction input>)
```

### Plugin help
Run the following command to get the plugin help string.

##### CLI
```shell script
mlflow deployments help -t torchserve
``` 


