# Deploying MNIST Handwritten Recognition using torchserve

## Package Requirement

Install the required packages using the following command

`pip install -r requirements.txt`

## Installing Deployment plugin

Run the following commands to install deployment plugin

`python setup.py build`
`python setup.py install`


## Generating model file (.pt)

Move to `mlflow-torchserve/examples/MNIST/` directory and run the following command to train the MNIST model  

Command: 
```
python mnist_model.py \
    --epochs 5 \
    --model-save-path "example2"
```

The trained model file `mnist_cnn.pt` and the sample input file `sample.json` will be saved into the model-save-path location(`example2` directory).

## Starting torchserve

create an empty directory `model_store` and run the following command to start torchserve.

`torchserve --start --model-store model_store`

## Creating and predict deployment

This example uses tensor as input for prediction.

In the previous example(example1), mlflow cli is used for predict. Input image path is provided as a cli argument.

Python plugin accepts 3 different types of input for prediction - Dataframe, Json and Tensor.

This example demonstrates input as tensor (refer `predict.py` where images are converted to tensor)

Run the following command to create and predict the output based on our test data - `test_data/one.png`

`python predict.py <PATH_TO_MODEL_FILE> <PATH_TO_HANDLER>>`

For ex:
`python predict.py /home/ubuntu/mlflow-torchserve/examples/MNIST/mnist_model.py mnist_handler.py`

Note: Torchserve plugin determines the version number by itself based on the deployment name. hence, version number 
is not a mandatory argument for the plugin. For example, by running `predict.py`, a new deployment `mnist_test` is created with version 1.

If needed, version number can also be explicitly mentioned as a config variable in `predict.py` as below.

```
config = {
    'MODEL_FILE': "<PATH_TO_MODEL_FILE>",
    'HANDLER': 'mnist_handler.py',
    'VERSION': 5.0

}
```  

MNIST model would predict the handwritten digit and the result will be printed in the console. 