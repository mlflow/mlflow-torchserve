# Deploying MNIST Handwritten Recognition using torchserve

## Training the model
The model is used to classify handwritten digits.
This example, autologs the trained model and its relevant parameters and metrics into mlflow using a single line of code. 
The example also illustrates how one can use the python plugin to deploy and test the model.
Python scripts `create_deployment.py` and `predict.py` have been used for this purpose.

Run the following command to train the MNIST model

CPU: `mlflow run . -P max_epochs=5`
GPU: `mlflow run . -P max_epochs=5 -P gpus=2 -P accelerator=ddp`

On the training completion, the MNIST model is stored as "model.pth" in current working directory.

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
2. serialized file path - `--serialized_file`
3. handler file path - `--handler`
4. model file path - `--model_file`

For example, to create another deployment the script can be triggered as

`python create_deployment.py --deployment_name mnist_deployment1`

## Predicting deployed model

To perform prediction, run the following script

`python predict.py`

The prediction results will be printed in the console. 

Following are the arguments which can be passed to predict_deployment script

1. deployment name - `--deployment_name"`
2. input file path - `--input_file_path`

For example, to perform prediction on the second deployment which we created. Run the following command

`python predict.py --deployment_name mnist_deployment1 --input_file_path test_data/one.png`
