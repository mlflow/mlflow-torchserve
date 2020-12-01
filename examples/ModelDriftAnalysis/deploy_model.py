import os
import time
import mlflow
import pandas as pd
import plotly.graph_objects as go
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import start_torchserve
import model_scoring
import predict
import argparse
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from pathlib import Path


def best_model_register(id, model_name, client):
    exp_list = client.search_runs(experiment_ids=id, run_view_type=ViewType.ACTIVE_ONLY)[0]
    best_run = dict(exp_list)
    artifact_uri = best_run.get("info").artifact_uri
    best_model_run_id = best_run.get("info").run_uuid
    mlflow.register_model(artifact_uri, model_name)
    return artifact_uri, best_model_run_id


def evaluate_deployed_model(test_data=None, block_size=0, model_file_path=None):
    predict.set_environment_variables("classifier.py", "logistic_handler.py")
    list_df = [test_data[i : i + block_size] for i in range(0, test_data.shape[0], block_size)]
    overall_test_results = []
    for i in range(len(list_df)):

        mar_file_name = "log_drift" + str(i)
        result = predict.create_deployment_and_predict(
            list_df[i], "class", "dates", model_file_path
        )
        result.rename(columns={0: "Predicted", 1: "Actual"}, inplace=True)
        accuracy_results = model_scoring.scorer(result)
        test_parameter_list = str.split(accuracy_results, "~")
        overall_test_results.append(
            (i, test_parameter_list[0], test_parameter_list[1], test_parameter_list[2])
        )
        if float(test_parameter_list[1]) < float(args.threshold_accuracy):
            break
    results_table = pd.DataFrame(
        overall_test_results, columns=["test_no", "Date", "Accuracy", "WrongPredictions"]
    )
    return results_table


def log_test_metrics_mlflow(run_id, test_results):
    with mlflow.start_run(run_id=run_id):
        for i in range(len(test_results)):
            mlflow.log_metric(
                "Avg_test_acc on -" + test_results["Date"][i], float(test_results["Accuracy"][i])
            )


def plot(dataframe):
    fig = go.Figure(
        data=[
            go.Bar(name="Accuracy", x=dataframe["Date"], y=dataframe["Accuracy"]),
            go.Bar(name="WrongPredictions", x=dataframe["Date"], y=dataframe["WrongPredictions"]),
        ]
    )
    fig.update_layout(
        barmode="group",
        title="Performance of The Deployed Model",
        xaxis_title="Tested Date",
        yaxis_title="Accuracy (Correct and Wrong Predictions)",
        legend_title="Predictions",
    )
    fig.show()


def check_degradation(result=None, degrade_index=0, threshold_accuracy=0):
    for i in range(len(result)):
        print(result["Accuracy"][i])
        if float(result["Accuracy"][i]) < threshold_accuracy:
            degrade_index = i
            break
    if degrade_index != None:
        print("Retrain the model with recent set of datapoints")
        start_torchserve.stop_torchserve()
        exit()
    else:
        print("no drift found for the given test dataset so far")
        start_torchserve.stop_torchserve()
        exit()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test_data_rows", default=100, help="Number of rows of test data to be used"
    )
    parser.add_argument("--block_size", default=20, help="It indicated batch size for test data")
    parser.add_argument(
        "--mlflow_experiment_name",
        help="Name of MLFLOW experiment in which results would be dumped",
    )
    parser.add_argument("--threshold_accuracy", help="KPI to be monitored")
    parser.add_argument(
        "--register_model_name", help="Name for the best model to be registered in MLFLOW"
    )

    args = parser.parse_args()

    if "MLFLOW_TRACKING_URI" in os.environ:
        tracking_uri = os.environ["MLFLOW_TRACKING_URI"]

    else:
        tracking_uri = "http://localhost:5000/"

    mlflow.tracking.set_tracking_uri(tracking_uri)

    client = MlflowClient(tracking_uri)
    model_name = "best logistic model for electricity demand classification"
    identifier = mlflow.get_experiment_by_name(args.mlflow_experiment_name)
    artifact_uri, run_id = best_model_register(
        identifier.experiment_id, args.register_model_name, client
    )
    start_torchserve.start_torchserve()
    time.sleep(5)
    df = pd.read_csv("new_demand.csv")
    test_data_rows = int(args.test_data_rows)
    model_test_data = df[int(0.8 * len(df)) : len(df)]
    test_data = model_test_data[0:test_data_rows]
    path = Path(_download_artifact_from_uri(artifact_uri))
    model_file_path = os.path.join(path, "model/data/model.pth")
    results = evaluate_deployed_model(
        test_data=test_data, block_size=int(args.block_size), model_file_path=model_file_path
    )
    log_test_metrics_mlflow(run_id, results)
    plot(results)
    degrade_index = None
    check_degradation(
        result=results,
        degrade_index=degrade_index,
        threshold_accuracy=float(args.threshold_accuracy),
    )
