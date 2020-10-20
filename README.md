# MLFLOW TORCHSERVE DEPLOYMENT PLUGIN

## Package Requirement

Following are the list of packages which needs to be installed before running the torchserve deployment plugin

1. torch-model-archiver
2. torchserve
3. mlflow

## Steps to build the package

Run the following commands to build the package

1. Build the plugin package ```python setup.py build```
2. Install the plugin package ```python setup.py install```
3. Deactivate the environment ```conda deactivate```
4. Activate the environment ```conda activate ENVIRONMENT_NAME```

## Sample Commands for deployment

1. Creating a new deployment - ```mlflow deployments create -t TARGET -m MODEL_URI --name DEPLOYMENT_NAME -C 'MODEL_FILE=MODEL_FILE_PATH' -C 'HANDLER_FILE=HANDLER_FILE_PATH'```
For Example: ```mlflow deployments create -t torchserve -m /home/ubuntu/deploy/linear.pt --name linear3  -C "MODEL_FILE=linear_model.py" -C "HANDLER_FILE=linear_handler.py"```

2. List all deployments - ```mlflow deployments list -t TARGET```
For Example: ```mlflow deployments list -t torchserve```

3. Get a deployment - ```mlflow deployments get -t TARGET --name DEPLOYMENT_NAME```
For Example: ```mlflow deployments get -t torchserve --name test```

4. Delete a deployment - ``mlflow deployments delete -t TARGET --name DEPLOYMENT_NAME``
For Example: ```mlflow deployments delete -t torchserve --name test2```

5. Update a deployment - ```mlflow deployments update -t TARGET -m MODEL_URI --name DEPLOYMENT_NAME```
For Example: ```mlflow deployments update -t torchserve -m /home/ubuntu/deploy/linear.pt --name test2```
