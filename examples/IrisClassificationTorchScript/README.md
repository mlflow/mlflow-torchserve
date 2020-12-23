# Deploying Iris Classification using torchserve

The code, adapted from this [repository](http://chappers.github.io/2020/04/19/torch-lightning-using-iris/),
is almost entirely dedicated to training, with the addition of a single mlflow.pytorch.autolog() call to enable automatic logging of params, metrics, and the TorchScript model.
TorchScript allows us to save the whole model locally and load it into a different environment, such as in a server written in
a completely different language.

## Training the model

To run the example via MLflow, navigate to the `examples/IrisClassificationTorchScript/` directory and run the command

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

After the training, we will convert the model to a TorchScript model using the function `torch.jit.script`.
At the end of the training process, scripted model is stored as `iris_ts.pt`

## Starting TorchServe

create an empty directory `model_store` and run the following command to start torchserve.

`torchserve --start --model-store model_store`

## Creating a new deployment

Run the following command to create a new deployment named `iris_test`

`mlflow deployments  create --name iris_test --target torchserve --model-uri iris_ts.pt -C "HANDLER=iris_handler.py"  -C "EXTRA_FILES=index_to_name.json"`

## Running prediction based on deployed model

Run the following command to invoke prediction of our sample input, where input.json is the sample input file and output.json stores the predicted outcome.

`mlflow deployments predict --name iris_test --target torchserve --input-path sample.json  --output-path output.json`
