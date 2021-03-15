#Titanic features attribution analysis using Captum and TorchServe.

In this example, we show how to use a pre-trained custom Titanic model to performing real time classification prediction(survived/Not survived) and features attributions with Captum and TorchServe. You can download the dataset from https://biostat.app.vumc.org/wiki/pub/Main/DataSets/titanic3.csv 

The inference service would return the prediction and attribution socre of features for a given test record.

### Running the code

To run the example via MLflow, navigate to the `examples/Titanic/` directory and run the command

```
mlflow run .

```

This will run `titanic_captum_interpret.py` with the default set of parameters such as `--max_epochs=100`. You can see the default value in the MLproject file.

In order to run the file with custom parameters, run the command

```
mlflow run . -P max_epochs=X
```

where X is your desired value for max_epochs.

If you have the required modules for the file and would like to skip the creation of a conda environment, add the argument --no-conda.

```
mlflow run . --no-conda
```

# Above commands will train the titanic model for further use.


# Serve a custom model on TorchServe

Run the commands given in following steps from the example dir. For example, if you cloned the repository into /home/my_path/serve, run the steps from /home/my_path/CaptumExample

 * Step - 1: Create a new model architecture file which contains model class extended from torch.nn.modules. In this example we have created [titanic model file](titanic.py).
 * Step - 2: Write a custom handler to run the inference on your model. In this example, we have added a [custom_handler](titanic_handler.py) which runs the inference on the input record using the above model and make prediction.
 * Step - 3: Create a torch model archive using the torch-model-archiver utility to archive the above files.
 
    ```bash
    torch-model-archiver --model-name titanic --version 1.0 --model-file titanic.py --serialized-file titanic_state_dict.pt  --handler  titanic_handler.py --extra-file index_to_name.json
    ```
   
 * Step - 5: Register the model on TorchServe using the above model archive file and run inference
   
    ```bash
    mkdir model_store
    mv titanic.mar model_store/
     torchserve --start --model-store model_store/ --ts-config config.properties
    curl http://127.0.0.1:8080/predictions/titanic --data '{"input_file_path" : "/home/ubuntu/titanic/test_data/titanic_survived.csv"}'
    ```
this prediction curl request give the prediction of (survived/not survived) for a given test record.

For captum Explanations on the Torchserve side, use the below curl request:

```bash
curl http://127.0.0.1:8080/explanations/titanic --data '{"input_file_path" : "/home/ubuntu/titanic/data/titanic_survived.csv"}'
```

this explanations curl request give us the average attribution for each feature. From the feature attribution information, we obtain some interesting insights regarding the importance of various features.
