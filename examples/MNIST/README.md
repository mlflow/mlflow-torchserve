# Deploying MNIST Handwritten Recognition using TorchServe

## Training the model

Follow the [respository](https://github.com/mlflow/mlflow/tree/master/examples/pytorch/MNIST/example1) to train MNIST handwritten digit recognition.
The above mentioned example, autologs the related parameters and model into mlflow using a single line of code.

## Installing Deployment plugin

Run the following commands to install deployment plugin

`python setup.py build`
`python setup.py install`

## Starting torchserve

create an empty directory `model_store` and run the following command to start torchserve.

`torchserve --start --model-store model_store`

## Creating and predict deployment

Python plugin accepts 3 different types of input for prediction - Dataframe, Json and Tensor.

This example demonstrates input as tensor (refer `predict.py` where images are converted to tensor)

Run the following command to create and predict the output based on our test data - `test_data/one.png`

`python predict.py <MODEL_URI>`

For ex: `python predict.py  s3://mlflow/artifacts/0/fe1407a3242c4f6f9a5748eca1ae9226/artifacts/model`

Note: Torchserve plugin determines the version number by itself based on the deployment name. hence, version number 
is not a mandatory argument for the plugin. For example, by running `predict.py`, a new deployment `mnist_test` is created with version 1.

Version number can also be explicitly mentioned as a config variable in `predict.py` as below.

```
config = {
    'HANDLER': 'mnist_handler.py',
    'VERSION': 1.0
}
```  

