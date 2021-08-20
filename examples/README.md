The examples in the folder illustrates training and deploying models using `mlflow-torchserve` plugin

1.`BertNewClassification` - Demonstrates a workflow using a pytorch based BERT model.
                            The example illustrates
                            <li> model training (fine tuning a pre-trained model) </li>
                            <li> logging of model,summary, parameters and extra files at the end of training    
                            <li> deployment of the  model in TorchServe </li>

2.`E2EBert` - Demonstrates a workflow using a pytorch-lightning based BERT model.  
                           The example illustrates,  
                           <li> model training (fine tuning a pre-trained model)  
                           <li> model saving and loading using mlflow autolog  
                           <li> deployment of the  model in TorchServe  
                           <li> Calculating explanations using captum  

3.`IrisClassification` -  Demonstrates distributed training (DDP) and deployment using Iris Dataset and MLflow-torchserve plugin. This example illustrates the use of model signature.

4.`Iris_TorchScript` -   Demonstrates saving TorchScript version of the Iris Classification model and deployment of the same using MLflow-torchserve plugin with `MLflow cli` commands  

5.`MNIST` - Demonstrates training of MNIST handwritten digit recognition and deployment of the model using MLflow-torchserve **python plugin** - This examples illustrates on saving the state dict using `mlflow.pytorch.save_state_dict` library.

6.`TextClassification` - Demonstrates training of TextClassification example and deployment of the model using using MLflow-torchserve **python plugin**  

7.`Titanic` - Demonstrates training of titanic dataset using pytorch and deploying the model using mlflow-torchserve plugin. This example illustrates the validation using `captum` library.