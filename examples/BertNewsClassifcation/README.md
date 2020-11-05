# Deploying Bert - News Classification using torchserve

## Package Requirement

Install the required packages using the following command

`pip install -r requirements.txt`

## Installing Deployment plugin

Run the following commands to install deployment plugin

`python setup.py build`
`python setup.py install`

## Generating model file (.pt)

This example illustrates, storing additional artifacts such as `requirements.txt` and bert vocabulary `bert_base_uncased_vocab.txt`
along with the model.

`requirements_file` and `extra_files` argument in `mlflow.pytorch.log_model` facilitates to store these additional artifacts along
with the model into mlflow.

This example performs news classification using Hugging Face bert model and requires `transformers` package to train and predict.
`trasnformers` package is added as a requirement in `requirements.txt` and stored along with the model using requirement argument.

Torchserve deployment plugin has the ability to detect and add these `requirements.txt` and the extra files. And hence, during the
mar file generation, torchserve automatically bundles the `requirements.txt`and extra files along with model.

Note: The arguments `requirements_file` and `extra_files` in `mlflow.pytorch.log_model` are optional.

Run the `news_classifier.py` script which will fine tune the model based on play store review comments. 

By default,  the script exports the model file as `bert_pytorch.pt` and generates a sample input file `sample.json`

Command: 
```
python news_classifier.py \
    --epochs 5 \
    --num-samples 50000 \
    --tracking-uri http://localhost:5000 \
    --model-save-path /home/ubuntu/mlflow-torchserve/examples/BertNewsClassification/models
```

This command will run the training process and exports the model to the `model-save-path` location.
Above command exports the mlflow model into `/home/ubuntu/mlflow-torchserve/examples/BertNewsClassification/models` directory. 

By default this example logs the requirements(`requirements.txt`) and vocabulary file(`bert_base_cased_vocab.txt`) along with the model.

## Starting torchserve

create an empty directory `model_store` and run the following command to start torchserve.

`torchserve --start --model-store model_store`

## Creating a new deployment

Run the following command to create a new deployment named `news_classification_test`

`mlflow deployments create -t torchserve -m file:///home/ubuntu/mlflow-torchserve/examples/BertNewsClassification/models --name news_classification_test -C "MODEL_FILE=news_classifier.py" -C "HANDLER=news_classifier_handler.py"`

Note: Torchserve plugin determines the version number by itself based on the deployment name. hence, version number 
is not a mandatory argument for the plugin. For example, the above command will create a deployment `mnist_test` with version 1.

If needed, version number can also be explicitly mentioned as a config variable.

`mlflow deployments create -t torchserve -m file:///home/ubuntu/mlflow-torchserve/examples/BertNewsClassification/models --name news_classification_test -C "VERSION=5.0" -C "MODEL_FILE=news_classifier.py" -C "HANDLER=news_classifier_handler.py"`


## Running prediction based on deployed model

For testing the fine tuned model, a sample input text is placed in `input.json`
Run the following command to invoke prediction of our sample input 

`mlflow deployments predict --name news_classification_test --target torchserve --input_path sample.json  --output_path output.json`

Bert model would predict the classification of the given news text and store the output in `output.json`.
