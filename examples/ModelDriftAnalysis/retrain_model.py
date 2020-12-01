import classifier
import pandas as pd
from ax.service.ax_client import AxClient
import mlflow.pytorch
import argparse
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import os


def model_training_hyperparameter_tuning(df, max_epochs, experiment_name, total_trials):

    if "MLFLOW_TRACKING_URI" in os.environ:
        tracking_uri = os.environ["MLFLOW_TRACKING_URI"]

    else:
        tracking_uri = "http://localhost:5000/"

    mlflow.tracking.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    client = MlflowClient(tracking_uri)

    identifier = mlflow.get_experiment_by_name(args.mlflow_experiment_name)

    runs = client.search_runs(
        experiment_ids=identifier.experiment_id, run_view_type=ViewType.ACTIVE_ONLY
    )[0]
    best_run = dict(runs)
    run_id = best_run.get("info").run_id
    print("run_id is ", run_id)
    mlflow.start_run(run_id=run_id, run_name="BaseModel")

    ax_client = AxClient()
    ax_client.create_experiment(
        parameters=[
            {"name": "lr", "type": "range", "bounds": [1e-3, 0.6], "log_scale": True},
            {"name": "weight_decay", "type": "range", "bounds": [1e-4, 1e-3]},
            {"name": "nesterov", "type": "choice", "values": [True, False]},
            {"name": "momentum", "type": "range", "bounds": [0.5, 1.0]},
        ],
        objective_name="test_accuracy",
    )

    total_trials = total_trials
    # child runs begin here.

    for i in range(total_trials):

        with mlflow.start_run(nested=True, run_name="Trial " + str(i)) as child_run:

            parameters, trial_index = ax_client.get_next_trial()
            dm = classifier.DataModule(df)

            # evaluate params
            model = classifier.Classification(kwargs=parameters)

            # calling the model
            test_accuracy = classifier.train_evaluate(
                parameterization=None, dm=dm, model=model, max_epochs=max_epochs
            )

            # completion of trial
            ax_client.complete_trial(trial_index=trial_index, raw_data=test_accuracy.item())

    best_parameters, metrics = ax_client.get_best_parameters()
    for param_name, value in best_parameters.items():
        mlflow.log_param("optimum " + param_name, value)

    mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_epochs",
        default=2,
        help="Describes the number of times a neural network has to be trained",
    )
    parser.add_argument(
        "--mlflow_experiment_name",
        help="Name of MLFLOW experiment in which results would be dumped",
    )
    parser.add_argument(
        "--total_trials",
        default=3,
        help="It indicated number of trials to be run for the optimization experiment",
    )
    args = parser.parse_args()
    df = pd.read_csv("new_demand.csv")
    train_dataset_length = int(len(df) * 0.82)
    model_data = df[0:train_dataset_length]
    model_test_data = df[train_dataset_length : len(df)]
    mlflow_experiment_name = "non_auto_drift"
    model_training_hyperparameter_tuning(
        model_data, int(args.max_epochs), args.mlflow_experiment_name, int(args.total_trials)
    )
