# Deploying Iris Classification using torchserve

The code, adapted from this [repository](http://chappers.github.io/2020/04/19/torch-lightning-using-iris/), 
is almost entirely dedicated to training, with the addition of a single mlflow.pytorch.autolog() call to enable automatic logging of params, metrics, and models.

## Package Requirement

Install the required packages using the following command

`pip install -r requirements.txt`
 
 
 ### Running the code
 
To run the example via MLflow, navigate to the `examples/IrisClassification/` directory and run the command

```
mlflow run .

```

This will run `iris_classification.py` with the default set of parameters such as `--max_epochs=100`. You can see the default value in the MLproject file.

In order to run the file with custom parameters, run the command

```
mlflow run . -P max_epochs=X
```

where X is your desired value for max_epochs.

If you have the required modules for the file and would like to skip the creation of a conda environment, add the argument --no-conda.

```
mlflow run . --no-conda
```

To run it in gpu, use the following command

```
mlflow run . -P gpus=2 -P accelerator="ddp"
```


## Installing Deployment plugin

Run the following commands to install deployment plugin

`python setup.py build`
`python setup.py install`

## Starting torchserve

create an empty directory `model_store` and run the following command to start torchserve.

`torchserve --start --model-store model_store`

## Creating a new deployment

Run the following command to create a new deployment named `iris_test`

`mlflow deployments  create --name iris_test --target torchserve --model-uri iris.pt -C "MODEL_FILE=iris_classification.py" -C "HANDLER=iris_handler.py" -C "EXTRA_FILES=index_to_name.json"`


## Running prediction based on deployed model

For testing [iris dataset](http://archive.ics.uci.edu/ml/datasets/Iris/), we are going to use a sample input tensor placed in `sample.json` file. 

Run the following command to invoke prediction of our sample input, whose output is stored in output.json file.

`mlflow deployments predict --name iris_test --target torchserve --input-path sample.json  --output-path output.json`


