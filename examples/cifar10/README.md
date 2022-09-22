# Deploying Cifar10 image classification using torchserve

This example demonstrates fine tuning of resnet model using cifar10 dataset.

Follow the link given below to set backend store

https://www.mlflow.org/docs/latest/tracking.html#storage

## Training the model

This example, autologs the trained model and its relevant parameters and metrics into mlflow using a single line of code. 
The example also illustrates how one can use the python plugin to deploy and test the model.
Python scripts `create_deployment.py` and `predict.py` have been used for this purpose.

Run the following command to train the cifar10 model

CPU: `mlflow run . -P max_epochs=5`
GPU: `mlflow run . -P max_epochs=5 -P devices=2 -P strategy=ddp -P accelerator=gpu`

At the end of the training, Cifar10 model will be saved as state dict (resnet.pth) in the current working directory

## Starting torchserve

create an empty directory `model_store` and run the following command to start torchserve.

`torchserve --start --model-store model_store`

## Creating a new deployment

This example uses image path as input for prediction.

To create a new deployment, run the following command

`python create_deployment.py`

It will create a new deployment named `cifar_test`.

Following are the arguments which can be passed to create_deployment script

1. deployment name - `--deployment_name`
2. path to serialized file - `--model_uri`
3. handler file path - `--handler`
4. model file path - `--model_file`

Note:
if the torchserve is running with a different "model_store" locations, the model-store path 
can be passed as input using `--export_path` argument.

For example:

`python create_deployment.py --deployment_name cifar_test1 --export_path /home/ubuntu/model_store`

## Predicting deployed model

To perform prediction, run the following script

`python inference.py`

The prediction results will be printed in the console. 

to save the inference output in file run the following command

`python inference.py --output_file_path prediction_result.json`

Following are the arguments which can be passed to predict_deployment script

1. deployment name - `--deployment_name"`
2. input file path - `--input_file_path`
3. path to write the result - `--output_file_path`


## Calculate captum explanations

To perform explain request, run the following script

`python inferene.py --inference_type explanation`

to save the explanation output in file run the following command

`python inference.py --inference_type explanation --output_file_path explanation_result.json`


## Viewing captum results

Use the notebook - `Cifar10_Captum.ipynb` to view the captum results.
