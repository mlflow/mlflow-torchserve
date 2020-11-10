In this example, we train a Pytorch Lightning model to classify iris based on height and width of sepal and petal. Then, we convert the model to TorchScript 
and serve the scripted model in TorchServe from Mlflow. 

## Training the model

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

After the training, we will convert the model to a TorchScript model using torch.jit.script.
At the end of the training process, scripted model is stored as `iris_ts.pt`

## Starting torchserve

create an empty directory `model_store` and run the following command to start torchserve.

`torchserve --start --model-store model_store`

## Creating a new deployment

Run the following command to create a new deployment named `iris_test`

`mlflow deployments  create --name iris_test --target torchserve --model-uri iris_ts.pt -C "MODEL_FILE=iris_classification.py" -C "HANDLER=iris_handler.py"`


## Running prediction based on deployed model

For testing iris dataset, we are going to use a sample input tensor placed in `sample.json` file. 
Run the following command to invoke prediction of our sample input

`mlflow deployments predict --name iris_test --target torchserve --input-path sample.json  --output-path output.json`

The deployed model would predict the type of iris and store the output in `output.json`.
