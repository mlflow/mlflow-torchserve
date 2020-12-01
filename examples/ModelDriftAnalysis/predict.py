from mlflow.deployments import get_deploy_client
import pandas as pd
import os


def set_environment_variables(model_file, handler_file):
    # os.environ['version']=version
    os.environ["model_file"] = model_file
    os.environ["HANDLER"] = handler_file


def create_deployment_and_predict(dataframe, dataframe_label, dataframe_datecolumn_name, model_uri):
    plugin = get_deploy_client("torchserve")
    config = {"MODEL_FILE": os.environ["model_file"], "HANDLER": os.environ["HANDLER"]}

    plugin.create_deployment(name="test", model_uri=model_uri, config=config)

    filt_df = dataframe.loc[:, dataframe.columns != dataframe_label]
    filt_df = filt_df.set_index(dataframe_datecolumn_name)
    listofdfrows = filt_df.to_numpy().tolist()

    tot_results = []

    for row in listofdfrows:

        df = pd.DataFrame([row])
        res = plugin.predict("test", df)
        tot_results.append(res)

    actual = dataframe[dataframe_label].tolist()
    for i in range(0, len(tot_results)):
        tot_results[i] = int(tot_results[i])
    predicted_results = pd.DataFrame(list(zip(tot_results, actual)))
    predicted_results.set_index(filt_df.index, inplace=True)
    plugin.delete_deployment("test")
    return predicted_results
