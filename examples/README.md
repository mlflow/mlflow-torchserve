This folder contains there different examples as shown below

1. `MNIST` - Demonstrates training of MNIST handwritten digit recognition and deployment of the model using MLflow-torchserve **python plugin**
2. `IrisClassification` - Demonstrates training of IrisClassification example
                          and deployment of the model using MLflow-torchserve plugin with `MLflow cli` commands.
3. `BertNewClassification` - This Bert news classification example covers End to End scenario. In this example, a hugging face bert model
                             is trained to classify the news texts and its parameters and artifacts are stored in MLflow using autologging technique.
                             MLflow-torchserve plugin is used to deploy the model from MLflow and sample predicts are made.