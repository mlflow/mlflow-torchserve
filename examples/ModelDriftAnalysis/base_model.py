import classifier
import pandas as pd
import mlflow
import argparse
import os


def model_training_hyperparameter_tuning(
    df, max_epochs, experiment_name, drift_version_count, total_trials, params
):
    dm = classifier.DataModule(df)
    model = classifier.Classification(kwargs=params)
    if "MLFLOW_TRACKING_URI" in os.environ:
        tracking_uri = os.environ["MLFLOW_TRACKING_URI"]

    else:
        tracking_uri = "http://localhost:5000/"

    mlflow.tracking.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)
    mlflow.start_run(run_name="BaseModel")
    mlflow.set_tag("Version", "Baseline")

    classifier.train_evaluate(parameterization=None, dm=dm, model=model, max_epochs=max_epochs)


def build_baseline_model(
    model_data=None,
    max_epochs=1,
    drift_version_count=0,
    mlflow_experiment_name=None,
    total_trials=0,
    params=None,
):
    model_training_hyperparameter_tuning(
        model_data, max_epochs, mlflow_experiment_name, drift_version_count, total_trials, params
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_epochs",
        default=2,
        help="Describes the number of times a neural network has to be trained",
    )
    parser.add_argument(
        "--mlflow_experiment_name",
        default="Default",
        help="Name of MLFLOW experiment in which results would be dumped",
    )
    args = parser.parse_args()
    df = pd.read_csv("new_demand.csv")
    model_training_data = df[0 : int(len(df) * 0.70)]
    model_test_data = df[int(len(df) * 0.70) : int(0.8 * len(df))]
    params = {"lr": 0.011, "momentum": 0.9, "weight_decay": 0, "nesterov": False}
    build_baseline_model(
        model_data=model_training_data,
        max_epochs=int(args.max_epochs),
        mlflow_experiment_name=args.mlflow_experiment_name,
        params=params,
    )
