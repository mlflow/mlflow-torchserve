The examples in the folder illustrates training and deploying models using `mlflow-torchserve` plugin

1.`BertNewClassification`-Demonstrates a workflow using a pytorch based BERT model.
                           The example illustrates,
                           a.) model training (fine tuning a pre-trained model), 
                           b.)logging of model,summary, parameters and extra files at the end of training
                           c.)deployment of the  model in TorchServe

2.`E2EBert`-Demonstrates a workflow using a pytorch-lightning based BERT model.
                           The example illustrates,
                           a.) model training (fine tuning a pre-trained model),
                           b.) model saving and loading using mlflow autolog
                           c.) deployment of the  model in TorchServe
                           d.) Calculating explanations using captum

3.`IrisClassification` -  Demonstrates distributed training (DDP) and deployment using Iris Dataset and MLflow-torchserve plugin.

4.`Iris_TorchScript` -   Demonstrates saving TorchScript version of the Iris Classification model and deployment of the same using MLflow-torchserve plugin with `MLflow cli` commands.

5.`MNIST` - Demonstrates training of MNIST handwritten digit recognition and deployment of the model using MLflow-torchserve **python plugin**

6.`TextClassification` - Demonstrates training of TextClassification example and deployment of the model using using MLflow-torchserve **python plugin**

7. `Titanic` - Demonstrates training of titanic dataset using pytorch and deploying the model using mlflow-torchserve plugin
