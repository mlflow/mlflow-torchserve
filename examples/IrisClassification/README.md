In this example, we train a Pytorch Lightning model to classify iris based on height and width of sepal and petal. 

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

To run it in gpu, use the following command

```
mlflow run . -P gpus=2 -P accelerator="ddp"
```

At the end of the training process, model state_dict is stored as `iris.pt`

## Starting torchserve

create an empty directory `model_store` and run the following command to start torchserve.

`torchserve --start --model-store model_store`

## Creating a new deployment

Run the following command to create a new deployment named `iris_test`

`mlflow deployments  create --name iris_test --target torchserve --model-uri iris.pt -C "MODEL_FILE=iris_classification.py" -C "HANDLER=iris_handler.py" -C "EXTRA_FILES=index_to_name.json"`


## Running prediction based on deployed model

IrisClassification model takes 4 different parameters - sepel length, sepel width, petal length and petal width and 
these parameters can be passed as tensor. For ex: `[4.4000, 3.0000, 1.3000, 0.2000]`

For more details on dataset, please refer to the URL - http://archive.ics.uci.edu/ml/datasets/Iris/

For testing iris dataset, we are going to use a sample input tensor placed in `sample.json` file. 


Run the following command to invoke prediction of our sample input

`mlflow deployments predict --name iris_test --target torchserve --input-path sample.json  --output-path output.json`

The model will classify the flower as one among these three types - `SETOSA` , `VERSICOLOR`, `VIRGINICA` and store it in `output.json`
