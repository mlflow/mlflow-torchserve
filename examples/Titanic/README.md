## Using Captum to interpret Pytorch models and serving on Torchserve

In this example, we will demonstrate the basic features of the [Captum](https://captum.ai/) interpretability,and serving the model on torchserve through an example model trained on the Titanic survival data. you can download the data from [titanic](https://biostat.app.vumc.org/wiki/pub/Main/DataSets/titanic3.csv)

We will first train a deep neural network on the data using PyTorch and use Captum to understand which of the features were most important and how the network reached its prediction.

you can get more details about used attributions methods used in this example

1. [Titanic_Basic_Interpret](https://captum.ai/tutorials/Titanic_Basic_Interpret)
2. [integrated-gradients](https://captum.ai/docs/algorithms#primary-attribution)
3. [layer-attributions](https://captum.ai/docs/algorithms#layer-attribution)
 

The inference service would return the prediction and  avg attribution socre of features for a given target for a input test record.

### Running the code

To run the example via MLflow, navigate to the `examples/Titanic/` directory and run the command

```
mlflow run .

```

This will run `titanic_captum_interpret.py` with default parameter values, e.g.  `--max_epochs=100` and `--use_pretrained_model False`. You can see the full set of parameters in the `MLproject` file within this directory.

In order to run the file with custom parameters, run the command

```
mlflow run . -P max_epochs=X
```

where X is your desired value for max_epochs.

If you have the required modules for the file and would like to skip the creation of a conda environment, add the argument --no-conda.

```
mlflow run . --no-conda
```

Above commands will train the titanic model for further use.


Serve a custom model on TorchServe

Run the commands given in following steps from the example dir. For example, run the steps from `examples/Titanic/`

 * Step - 1: Create a new model architecture file which contains model class extended from torch.nn.modules. In this example we have created [titanic model file](titanic.py).
 * Step - 2: Write a custom handler to run the inference on your model. In this example, we have added a [custom_handler](titanic_handler.py) which runs the inference on the input record using the above model and make prediction.
 * Step - 3: Create a torch model archive using the torch-model-archiver utility to archive the above files.
 
    ```bash
    torch-model-archiver --model-name titanic --version 1.0 --model-file titanic.py --serialized-file models/titanic_state_dict.pt  --handler  titanic_handler.py --extra-file index_to_name.json
    ```
   
 * Step - 5: Register the model on TorchServe using the above model archive file and run inference. You need to modify/pass the path of the CSV in the input.json
   
    ```bash
    mkdir model_store
    mv titanic.mar model_store/
     torchserve --start --model-store model_store/ --ts-config config.properties
    curl -H "Content-Type: application/json" http://127.0.0.1:8080/predictions/titanic --data @test_data/input.json
    ```
this prediction curl request give the prediction of (survived/not survived) for a given test record.

For captum Explanations on the Torchserve side, use the below curl request:

```bash
curl -H "Content-Type: application/json"  http://127.0.0.1:8080/explanations/titanic --data @test_data/input.json

```

this explanations curl request give us the average attribution for each feature. From the feature attribution information, we obtain some interesting insights regarding the importance of various features.
