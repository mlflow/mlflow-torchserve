# Deploying Text Classification model

Download `train.py` and `model.py` from the [respository](https://github.com/pytorch/serve/tree/master/examples/text_classification)
and subsequently run the following command to train the model in either CPU/GPU.

CPU: `python train.py AG_NEWS --device cpu --save-model-path  model.pt --dictionary source_vocab.pt`

GPU: `python train.py AG_NEWS --device cuda --save-model-path  model.pt --dictionary source_vocab.pt`

At the end of the training, model file `model.pt` and vocabulary file `source_vocab.pt` will be stored in the current directory.

## Starting TorchServe

create an empty directory `model_store` and run the following command to start torchserve.

`torchserve --start --model-store model_store`

## Creating a new deployment

This example uses the default TorchServe text handler to generate the mar file.

To create a new deployment, run the following command

`python create_deployment.py --deployment_name text_classification --model_file model.py   --serialized_file model.pt  --extra_files "source_vocab.pt,index_to_name.json"`

## Predicting deployed model

To perform prediction, run the following script

`python predict.py`

The prediction results will be printed in the console.
