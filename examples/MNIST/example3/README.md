# Deploying MNIST Handwritten Recognition using torchserve

## Package Requirement

Install the required packages using the following command

`pip install -r requirements.txt`

## Installing Deployment plugin

Run the following commands to install deployment plugin

`python setup.py build`
`python setup.py install`


## Generating model file (.pt)

Run the `mnist_model.py` script which will perform training on MNIST handwritten dataset. 

This example primarily focuses on logging and using additional/supporting files during deployment. 
`number_to_text.json` file present in this example has the output mappings. 

The python package requirements are listed in `requirements.txt`.
This example logs `requirements.txt` as an additional artifact. Torchserve plugin auto detects the
requirements file and adds it as the `-r` argument in `torch-model-archiver` during the `create_deployment` process

Command: 

```
python mnist_model.py \
    --epochs 5 \
    --model-save-path "example3"
    --save-mlflow-model True
```

This command will run the training process and exports the model to the `model-save-path` location.
Above command exports the mlflow model into `mlflow-torchserve/examples/MNIST/example3/models` directory. 

## Starting torchserve

create an empty directory `model_store` and run the following command to start torchserve.

`torchserve --start --model-store model_store`

## Creating and predict deployment

Run the following command to create a new deployment named `mnist_test`.

`mlflow deployments  create --name mnist_test --target torchserve --model-uri file://<PATH_TO_SAVED_MODEL>> -C "MODEL_FILE=<PATH_TO_MODEL_FILE>" -C "HANDLER=mnist_handler.py"`

For Example:
`mlflow deployments  create --name mnist_test --target torchserve --model-uri file:///home/ubuntu/mlflow-torchserve/examples/MNIST/example3/models -C "MODEL_FILE=/home/ubuntu/mlflow-torchserve/examples/MNIST/mnist_model.py" -C "HANDLER=mnist_handler.py"`

Note: Torchserve plugin generates the version number based on the deployment name. hence, version number 
is not a mandatory argument for the plugin. For example, the above command will create a deployment `mnist_test` with version 1.

If needed, version number can also be explicitly mentioned as a config variable.

`mlflow deployments  create --name mnist_test --target torchserve --model-uri file:///home/ubuntu/mlflow-torchserve/examples/MNIST/example3/models -C "VERSION=5.0" -C "MODEL_FILE=/home/ubuntu/mlflow-torchserve/examples/MNIST/mnist_model.py" -C "HANDLER=mnist_handler.py"`

## Running prediction based on deployed model

For testing Handwritten dataset, we are going to use a sample image placed in `test_data` directory. 
Run the following command to invoke prediction of our sample input `test_data/one.png`

`mlflow deployments predict --name mnist_test --target torchserve --input-path sample.json  --output-path output.json`

Torchserve downloads the entire model directory along with the extra artifacts. 

In the handler file, the extra files are downloaded and results are mapped based on the json files - `number_to_text.json`

MNIST model would predict the handwritten digit and store the output as `one` in `output.json`.
