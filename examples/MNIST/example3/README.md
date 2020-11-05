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
requirments file and adds it as the `-r` argument in `torch-model-archiver` during the `create_deployment` process

The mapping file is pushed into mlflow along with the model using `mlflow.pytorch` library. 

Command: 

```
python mnist_model.py \
    --epochs 5 \
    --batch-size 64 \
    --lr 0.01 \
    --model-save-path /home/ubuntu/mlflow-torchserve/examples/MNIST/example3/models
```

This command will run the training process and exports the model to the `model-save-path` location.
Above command exports the mlflow model into `/home/ubuntu/mlflow-torchserve/examples/MNIST/example3/models` directory. 

## Starting torchserve

create an empty directory `model_store` and run the following command to start torchserve.

`torchserve --start --model-store model_store`

## Creating and predict deployment

Run the following command to create a new deployment named `mnist_test`.

Since the model is exported to mlflow, set the --model-uri to the S3 path where model is logged. 

`mlflow deployments  create --name mnist_test --target torchserve --model-uri file:///home/ubuntu/mlflow-torchserve/examples/MNIST/example3/models -C "MODEL_FILE=mnist_model.py" -C "HANDLER=mnist_handler.py"`

Note: Torchserve plugin determines the version number by itself based on the deployment name. hence, version number 
is not a mandatory argument for the plugin. For example, the above command will create a deployment `mnist_test` with version 1.

If needed, version number can also be explicitly mentioned as a config variable.

`mlflow deployments  create --name mnist_test --target torchserve --model-uri file:///home/ubuntu/mlflow-torchserve/examples/MNIST/example3/models -C "VERSION=5.0" "-C "MODEL_FILE=mnist_model.py" -C "HANDLER=mnist_handler.py"`


## Running prediction based on deployed model

For testing Handwritten dataset, we are going to use a sample image placed in `test_data` directory. 
Run the following command to invoke prediction of our sample input `test_data/one.png`

`mlflow deployments predict --name mnist_test --target torchserve --input_path sample.json  --output_path output.json`

Torchserve downloads the entire model directory along with the extra artifacts. 

In the handler file, the extra files are downloaded and results are mapped based on the json files - `number_to_text.json`

MNIST model would predict the handwritten digit and store the output as `one` in `output.json`.
