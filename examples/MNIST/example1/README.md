# Deploying MNIST Handwritten Recognition using torchserve

## Package Requirement

Install the required packages using the following command

`pip install -r requirements.txt`

## Installing Deployment plugin

move to `mlflow/pytorch/torchserve` and run the following commands to install deployment plugin

`python setup.py build`
`python setup.py install`

## Generating model file (.pt)

Run the `mnist_model.py` script which will perform training on MNIST handwritten dataset. 

By default,  the script exports the model file as `model_cnn.pt` and generates a sample input file `sample.json'

Command: `python mnist_model.py --epochs 5`

## Starting torchserve

create an empty directory `model_store` and run the following command to start torchserve.

`torchserve --start --model-store model_store`

## Creating a new deployment

Run the following command to create a new deployment named `mnist_test`

`mlflow deployments create -t torchserve -m mnist_cnn.pt --name mnist_test -C "MODEL_FILE=mnist_model.py" -C "HANDLER_FILE=mnist_handler.py"`

## Running prediction based on deployed model

For testing Handwritten dataset, we are going to use a sample image placed in `test_data` directory. 
Run the following command to invoke prediction of our sample input `test_data/one.png`

`mlflow deployments predict --name mnist_test --target torchserve --input_path sample.json  --output_path output.json`

MNIST model would predict the handwritten digit and store the output in `output.json`.
