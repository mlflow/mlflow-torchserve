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

## Sample Commands for deployment

1. Creating a new deployment - `mlflow deployments create -t TARGET -m MODEL_URI --name DEPLOYMENT_NAME -C 'MODEL_FILE=MODEL_FILE_PATH' -C 'HANDLER_FILE=HANDLER_FILE_PATH'` \
For Example: ```mlflow deployments create -t torchserve -m linear.pt --name linear  -C "MODEL_FILE=linear_model.py" -C "HANDLER_FILE=linear_handler.py"```

2. List all deployments - ```mlflow deployments list -t TARGET``` \
For Example: ```mlflow deployments list -t torchserve```

3. Get a deployment - ```mlflow deployments get -t TARGET --name DEPLOYMENT_NAME``` \
For Example: 
```mlflow deployments get -t torchserve --name linear``` - Gets the details of the default version of the model \
```mlflow deployments get -t torchserve --name linear/3.0``` - Gets the detals of linear model version 3.0 \
```mlflow deployments get -t torchserve --name linear/all``` - Gets the details of all the versions of the same model \

4. Delete a deployment - ``mlflow deployments delete -t TARGET --name DEPLOYMENT_NAME`` \
For Example: ```mlflow deployments delete -t torchserve --name linear/2.0``` \

5. Update a deployment - ```mlflow deployments update -t TARGET -m MODEL_URI --name DEPLOYMENT_NAME``` \
For Example: \
`mlflow deployments update -t torchserve -m linear.pt --name "linear" -C "min-worker=2"` - Updates default version of the model with 2 workers \
`mlflow deployments update -t torchserve -m linear.pt --name "linear/2.0" -C "min-worker=2"` - Updates linear version "2.0" model with 2 workers \
`mlflow deployments update -t torchserve -m linear.pt --name "linear/2.0" -C "set-default=true"` - Sets 2.0 as the default version \

6. Predict deployment - ```mlflow deployments predict -t TARGET --name DEPLOYMENT_NAME --input-path INPUT_PATH``` \
For Example: \
```mlflow deployments predict -t torchserve --name "linear" --input-path input.json``` - Predicts based on default version of the model \
```mlflow deployments predict -t torchserve --name "linear/2.0" --input-path input.json``` - Predicts based on linear version "2.0" \
```mlflow deployments predict -t torchserve --name "linear" --input-path input.json --output-path output.json``` - Predicts based on default version and writes the result into output.json

