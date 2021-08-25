# Deploying MNIST Handwritten Recognition using torchserve

This example requires Backend store to be set for mlflow.

Follow the link given below to set backend store

https://www.mlflow.org/docs/latest/tracking.html#storage

## Training the model
The model is used to classify handwritten digits.
This example, autologs the trained model and its relevant parameters and metrics into mlflow using a single line of code. 
The example also illustrates how one can use the python plugin to deploy and test the model.
Python scripts `create_deployment.py` and `predict.py` have been used for this purpose.

Run the following command to train the MNIST model

CPU: `mlflow run . -P max_epochs=5`
GPU: `mlflow run . -P max_epochs=5 -P gpus=2 -P accelerator=ddp`

At the end of the training, MNIST model will be saved as state dict in the current working directory

## Deploying in remote torchserve instance

To deploy the model in remote torchserve instance follow

the steps in [remote-deployment.rst](../../docs/remote-deployment.rst) under `docs` folder.


## Deploying in local torchserve instance

## Starting torchserve

create an empty directory `model_store` and run the following command to start torchserve.

`torchserve --start --model-store model_store`

## Creating a new deployment

This example uses image path as input for prediction.

To create a new deployment, run the following command

`python create_deployment.py`

It will create a new deployment named `mnist_classification`.

Following are the arguments which can be passed to create_deployment script

1. deployment name - `--deployment_name`
2. registered mlflow model uri - `--registered_model_uri`
3. handler file path - `--handler`
4. model file path - `--model_file`

For example, to create another deployment the script can be triggered as

`python create_deployment.py --deployment_name mnist_deployment1`

Note:

if the torchserve is running with a different "model_store" locations, the model-store path 
can be passed as input using `--export_path` argument.

For example:

`python create_deployment.py --deployment_name mnist_deployment1 --export_path /home/ubuntu/model_store`

## Predicting deployed model

To perform prediction, run the following script

`python predict.py`

The prediction results will be printed in the console. 

Following are the arguments which can be passed to predict_deployment script

1. deployment name - `--deployment_name"`
2. input file path - `--input_file_path`

For example, to perform prediction on the second deployment which we created. Run the following command

`python predict.py --deployment_name mnist_deployment1 --input_file_path test_data/one.png`
