# Deploying BERT - News Classification using TorchServe

The code, adapted from this [repository](https://github.com/maknotavailable/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling.py),
is almost entirely dedicated to model training (fine tuning). Pytorch model including the extra files such as the vocabulary file and class mapping file, which are essential to make the model functional
are saved locally using the function `mlflow.pytorch.save_model`. By default,  the script exports the model file as `bert_pytorch.pt` and generates a sample input file `input.json`.
This example workflow includes the following steps,
1. A pre trained Hugging Face bert model is fine-tuned to classify news.
2. Model is saved with extra files model,summary, parameters and extra files at the end of training
3. Deployment of the  model in TorchServe.

Torchserve deployment plugin has the ability to detect and add the `requirements.txt` and the extra files. And hence, during the
mar file generation, TorchServe automatically bundles the `requirements.txt`and extra files along with the model.



## Package Requirement

Install the required packages using the following command

`pip install -r requirements.txt`


### Running the code
To run the example via MLflow, navigate to the `mlflow-torchserve/examples/BertNewsClassification_E2E` directory and run the command

```
mlflow run .
```

This will run `news_classifier.py` with the default set of parameters such as  `--max_epochs=5`. You can see the default value in the `MLproject` file.

In order to run the file with custom parameters, run the command

```
mlflow run . -P max_epochs=X
```

where `X` is your desired value for `max_epochs`.

If you have the required modules for the file and would like to skip the creation of a conda environment, add the argument `--no-conda`.

```
mlflow run . --no-conda

```

Note: The arguments `requirements_file` and `extra_files` in `mlflow.pytorch.log_model` are optional.

Run the `news_classifier.py` script which will fine tune the model based on the news dataset.

By default,  the script exports the model file as `bert_pytorch.pt` and generates a sample input file `input.json`

Command:

### Passing custom training parameters

The parameters can be overridden via the command line:

1. max_epochs - Number of epochs to train models. Training can be interrupted early via Ctrl+C
2. num_samples -Number of input samples required for training


For example:
```
mlflow run . -P max_epochs=5 -P num_samples=50000
```

Or to run the training script directly with custom parameters:
```
python news_classifier.py \
    --max_epochs 5 \
    --num_samples 50000 \
    --model_save_path /home/ubuntu/mlflow-torchserve/examples/BertNewsClassification/models
```

## Starting TorchServe

create an empty directory `model_store` and run the following command to start torchserve.

`torchserve --start --model-store model_store`

## Creating a new deployment

Run the following command to create a new deployment named `news_classification_test`

`mlflow deployments create -t torchserve -m file:///home/ubuntu/mlflow-torchserve/examples/BertNewsClassification/models --name news_classification_test -C "MODEL_FILE=news_classifier.py" -C "HANDLER=news_classifier_handler.py"`


Note: Torchserve plugin determines the version number by itself based on the deployment name. hence, version number
is not a mandatory argument for the plugin. For example, the above command will create a deployment `news_classification_test` with version 1.

If needed, version number can also be explicitly mentioned as a config variable.


`mlflow deployments create -t torchserve -m file:///home/ubuntu/mlflow-torchserve/examples/BertNewsClassification/models --name news_classification_test -C "VERSION=1.0" -C "MODEL_FILE=news_classifier.py" -C "HANDLER=news_classifier_handler.py"`


## Running prediction based on deployed model

The deployed BERT model would predict the classification of the given news text and store the output in `output.json`. Run the following command to invoke prediction of our sample input (input.json)

`mlflow deployments predict --name news_classification_test --target torchserve --input-path input.json  --output-path output.json`































