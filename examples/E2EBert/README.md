
# An End-2-End Deep Learning Workflow with BERT PreTrained Model

An `end-2-end` workflow describing how model training is done,followed by storing all the relevant information leading to model deployment and
testing. A pretrained BERT model is used to illustrate the workflow.

## Finetuning the BERT Pretrained Model
The code, adapted from this [repository](https://github.com/maknotavailable/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling.py), 
is almost entirely dedicated to model training, with the addition of a single mlflow.pytorch.autolog() call to enable automatic logging of params, metrics, and models,
including the extra files, followed by saving the finetuned model along with extra artifact files such as the vocabulary file and class mapping file, which are essential to make the model 
work and help in transforming the model outputs into corresponding labels respectively.

## Package Requirement

Ensure to install the `mlflow-torchserve` [prerequisites](https://github.com/mlflow/mlflow-torchserve#prerequisites) and 
[package](https://github.com/mlflow/mlflow-torchserve#installation) in your current python environment before starting.

Install the required packages using the following command

`pip install -r requirements.txt`
 
 
 ### Running the code
To run the example via MLflow, navigate to the `mlflow-torchserve/examples/E2EBert` directory and run the command

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

To run it in gpu, use the following command

```
mlflow run . -P gpus=2 -P accelerator="ddp"
```

Run the `news_classifier.py` script which will fine tune the model based on news dataset. 

By default,  the script exports the model file as `state_dict.pth` and generates a sample input file `input.json`



### Passing custom training parameters

The parameters can be overridden via the command line:

1. max_epochs - Number of epochs to train model. Training can be interrupted early via Ctrl+C
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
```

## Starting TorchServe

Create an empty directory `model_store` and run the following command to start TorchServe.

`torchserve --start --model-store model_store`

## Creating a new deployment

Run the following command to create a new deployment named `news_classification_test`

`mlflow deployments create -t torchserve -m state_dict.pth --name news_classification_test -C "MODEL_FILE=news_classifier.py" -C "HANDLER=news_classifier_handler.py" -C "EXTRA_FILES=class_mapping.json,bert_base_uncased_vocab.txt,wrapper.py"`

Note: TorchServe plugin determines the version number by itself based on the deployment name. hence, version number 
is not a mandatory argument for the plugin. For example, the above command will create a deployment `news_classification_test` with version 1.

If needed, version number can also be explicitly mentioned as a config variable.

`mlflow deployments create -t torchserve -m state_dict.pth --name news_classification_test -C "VERSION=1.0" -C "MODEL_FILE=news_classifier.py" -C "HANDLER=news_classifier_handler.py" -C "EXTRA_FILES=class_mapping.json,bert_base_uncased_vocab.txt,wrapper.py"`


## Running prediction and explain based on deployed model

For testing the fine tuned model, a sample input text is placed in `input.json`

Run the following command to invoke prediction of our sample input 

`mlflow deployments predict --name news_classification_test --target torchserve --input-path input.json  --output-path output.json`

Run the following command to invoke explain of our sample input 


`mlflow deployments explain --name news_classification_test --target torchserve --input-path input.json  --output-path output.json`

All the captum Insights visualization can be seen in the jupyter notebook added in this example