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
    --model-save-path "example1"
```

The trained model file `mnist_cnn.pt` and the sample input file `sample.json` will be saved into the model-save-path location(`example1` directory).

## Starting torchserve

create an empty directory `model_store` and run the following command to start torchserve.

`torchserve --start --model-store model_store`

## Creating a new deployment

Run the following command to create a new deployment named `mnist_test`

`mlflow deployments create -t torchserve -m mnist_cnn.pt --name mnist_test -C "MODEL_FILE=<BASE_PATH>/mlflow-torchserve/examples/MNIST/mnist_model.py" -C "HANDLER=mnist_handler.py"`

Note: replace Base Path with the actual of the mlflow-torchserve repository. 
For ex: `"/home/ubuntu/mlflow-torchserve/examples/MNIST/mnist_model.py"`

Note: Torchserve plugin determines the version number by itself based on the deployment name. hence, version number 
is not a mandatory argument for the plugin. For example, the above command will create a deployment `mnist_test` with version 1.

If needed, version number can also be explicitly mentioned as a config variable.

`mlflow deployments create -t torchserve -m mnist_cnn.pt --name mnist_test -C "VERSION=5.0" -C "MODEL_FILE=<BASE_PATH>/mlflow-torchserve/examples/MNIST/mnist_model.py" -C "HANDLER=mnist_handler.py"`     

## Running prediction based on deployed model

For testing Handwritten dataset, we are going to use a sample image placed in `test_data` directory. 
Run the following command to invoke prediction of our sample input `test_data/one.png`

`mlflow deployments predict --name mnist_test --target torchserve --input-path sample.json  --output-path output.json`

MNIST model would predict the handwritten digit and store the output in `output.json`.
