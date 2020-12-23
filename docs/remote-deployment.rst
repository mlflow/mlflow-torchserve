.. _remote deployment:

==============================================
Steps for deploying model in Remote TorchServe
==============================================

Steps to be done in remote:
===========================

Start the torchserve instance in remote server

.. code-block:: 

    torchserve --start --model-store model_store --ts-config config.properties

Ignore this step if torchserve is already running on the remote server.

Steps to be done in local:
==========================

Verify Remote TorchServe Connectivity:
--------------------------------------

Run the following command to test the remote torchserve connectivity.

.. code-block:: 

    curl "http://<REMOTE_TORCHSERVE_IP>:8081/models"

This command should retrieve the list of existing models present in remote torchserve. This is an optional step. If you are sure about the connectivity proceed with the deployment steps as stated below.. 

Training the model:
-------------------

Clone the code from - `https://github.com/mlflow/mlflow-torchserve <https://github.com/mlflow/mlflow-torchserve>`_ and move to `mlflow-torchserve/examples/MNIST` folder.

Train the model by running the following command:

.. code-block::

    mlflow run . -P registration_name=mnist_classifier

The script will train the model and at the end of the training process, the model is registered into mlflow as `mnist_classifier`.


Setting up Config Properties:
-----------------------------

Set the management and inference URL in config properties file. `config.properties` file is placed in the home directory of the repository.

For example:

.. code-block::

    inference_address=http://<REMOTE_TORCHSERVE_IP>:8080
    management_address=http://<REMOTE_TORCHSERVE_IP>:8081

Setting Environment variable:
-----------------------------

Once the config properties file is updated with the remote TorchServe instance details. Set the environment variable CONFIG_PROPERTIES  with the path of the config.properties file.

For example:

.. code-block::

    export CONFIG_PROPERTIES=/home/ubuntu/mlflow-torchserve/config.properties

Install mlflow-torchserve Plugin:
---------------------------------

Ignore this step if the mlflow-torchserve plugin is already installed.

Install torchserve plugin using the following command

.. code-block::

    pip install mlflow-torchserve

Creating a new deployment:
--------------------------


Run the following script to start with the deployment process.

.. code-block::

    python create_deployment.py --deployment_name mnist_test --registered_model_uri models:/mnist_classifier/1

This comment will generate a `mnist_test.mar` file inside the `model_store` folder.

Since, the model needs to be deployed on the remote torchserve, the mar file needs to be exposed as a public url. 

Here is an example of hosting the mar file using python http server and ngrok. Any alternate mechanism can be used to expose the mar file as public url (For ex: uploading it into a S3 bucket and assigning necessary permissions to download it from http/https url).

Start the http server from model store as below

.. code-block::

    cd model_store
    python -m http.server

This is to host the file in the local instance. 
The verification can be done by downloading the file from the browser or from terminal using wget.

Open the browser and hit - `http://localhost:8000/mnist_test.mar <http://localhost:8000/mnist_test.mar>`_

Or in the terminal do `wget` `http://localhost:8000/mnist_test.mar <http://localhost:8000/mnist_test.mar>`_

The mnist.mar file will be downloaded. However, remote torchserve instance, doesnt understand the mar file hosted in localhost. 

Download and unzip ngrok file from the following url - `https://ngrok.com/download <https://ngrok.com/download>`_

Run the following command to run ngrok -

.. code-block::

    ./ngrok http 8000

Copy the web address from the forwarding section and update the EXPORT_URL parameter in config.properties file.

For example:

.. code-block::

    inference_address=http://<REMOTE_TORCHSERVE_IP>:8080
    management_address=http://<REMOTE_TORCHSERVE_IP>:8081
    export_url= http://eda154810618.ngrok.io


Download the mar file using ngrok url . Open browser and hit

.. code-block::

    http://eda154810618.ngrok.io/mnist_test.mar

mnist_test.mar file should be downloaded.

We are all set for performing registration. To register the model in remote torchserve instance run

.. code-block::

    python register.py --mar_file_name mnist_test.mar

The plugin will download the mar file from ngrok url and register the model in the remote TorchServe instance.


.. code-block::

    mlflow deployments list -t torchserve

This command will list the mnist_model which is registered in a remote TorchServe instance.

Prediction:
-----------

The model is registered in the remote TorchServe instance and ready for prediction. For running sample prediction invoke the prediction script as below

.. code-block::

    python predict.py --deployment_name mnist_test

Prediction result “ONE” will be displayed in the console.


