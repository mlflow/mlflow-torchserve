#Titanic features attribution analysis using Captum and TorchServe.

In this example, we will demonstrate the basic features of the [Captum](https://captum.ai/) interpretability,and serving the model on torchserve through an example model trained on the Titanic survival data. you can download the data from [titanic](https://biostat.app.vumc.org/wiki/pub/Main/DataSets/titanic3.csv)

We will first train a deep neural network on the data using PyTorch and use Captum to understand which of the features were most important and how the network reached its prediction.

you can get more details about used attributions methods used in this example

1. [Titanic_Basic_Interpret](https://captum.ai/tutorials/Titanic_Basic_Interpret)
2. [integrated-gradients](https://captum.ai/docs/algorithms#primary-attribution)
3. [layer-attributions](https://captum.ai/docs/algorithms#layer-attribution)
 

The inference service would return the prediction and  avg attribution socre of features for a given target for a input test record.

### Running the code

To run the example via MLflow, navigate to the `examples/Titanic/` directory and run the commands

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

 * Step - 1: Create a new model architecture file which contains model class extended from torch.nn.modules. In this example we have created [titanic model file](titanic.py).
 * Step - 2: Write a custom handler to run the inference on your model. In this example, we have added a [custom_handler](titanic_handler.py) which runs the inference on the input record using the above model and make prediction.
 * Step - 3: Create an empty directory model_store and run the following command to start torchserve.
 
    ```bash
    torchserve --start --model-store model_store/ --ts-config config.properties
    ```
   
## Creating a new deployment
 Run the following command to create a new deployment named `titanic`

The `index_to_name.json` file is the mapping file, which will convert the discrete output of the model to one of the class (survived/not survived)
based on the predefined mapping.

```bash
mlflow deployments create --name titanic --target torchserve --model-uri models/titanic_state_dict.pt -C "MODEL_FILE=titanic.py" -C "HANDLER=titanic_handler.py" -C "EXTRA_FILES=index_to_name.json"
```

## Running prediction based on deployed model

For testing, we are going to use a sample test record placed in test_data folder in input.json 

Run the following command to invoke prediction on test record, whose output is stored in output.json file.

`mlflow deployments predict --name titanic --target torchserve --input-path test_data/input.json  --output-path output.json`

This model will classify the test record as survived or not survived and store it in `output.json`


Run the below command to invoke explain for feature importance attributions on test record. It will save the attribution image attributions_imp.png in test_data folder.

` mlflow deployments explain -t torchserve --name titanic --input-path  test_data/input.json`

this explanations command give us the average attribution for each feature. From the feature attribution information, we obtain some interesting insights regarding the importance of various features.
