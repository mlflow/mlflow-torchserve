# Deploying Text Classification model

## Installing Deployment plugin

Run the following commands to install deployment plugin

`python setup.py build`
`python setup.py install`


## Training the model


https://github.com/pytorch/serve/tree/master/examples/text_classification

Download `train.py` and `model.py` from above mentioned link and run the following command to train the model

CPU: `python train.py AG_NEWS --device cpu --save-model-path  model.pt --dictionary source_vocab.pt`

GPU: `python train.py AG_NEWS --device cuda --save-model-path  model.pt --dictionary source_vocab.pt`

At the end of the training model file `model.pt` and vocabulary file `source_vocab.pt` will be stored into current directory.

## Starting torchserve

create an empty directory `model_store` and run the following command to start torchserve.

`torchserve --start --model-store model_store`

## Creating a new deployment

This example takes text as input and the sample text is placed in `sample_text.txt`

To create a new deployment, run the following command

`python create_deployment.py`

It will create a new deployment named `text_classification`.

Following are the arguments which can be passed to create_deployment script

1. deployment name - `deployment_name`
2. serialized file path - `serialized_file`
3. handler file path - `handler`
4. model file path - `model_file_path`

For example, to create another deployment the script can be triggered as

`python create_deployment.py --deployment_name text_deployment_1`

## Predicting deployed model


To perform prediction, run the following script

`python predict.py`

The prediction results will be printed in the console. 



