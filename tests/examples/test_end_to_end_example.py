import mlflow
import pytest
import os
from mlflow.utils import process


@pytest.mark.usefixtures("start_torchserve")
def test_mnist_example():
    os.environ["MKL_THREADING_LAYER"] = "GNU"
    home_dir = os.getcwd()
    mnist_dir = "examples/MNIST"
    example_command = ["python", "mnist_model.py", "--max_epochs", "1"]
    process.exec_cmd(example_command, cwd=mnist_dir)

    assert os.path.exists(os.path.join(mnist_dir, "models", "state_dict.pth"))
    create_deployment_command = [
        "python",
        "create_deployment.py",
        "--export_path",
        os.path.join(home_dir, "model_store"),
    ]

    process.exec_cmd(create_deployment_command, cwd=mnist_dir)

    assert os.path.exists(os.path.join(home_dir, "model_store", "mnist_classification.mar"))

    predict_command = ["python", "predict.py"]
    res = process.exec_cmd(predict_command, cwd=mnist_dir)
    assert "ONE" in res[1]


@pytest.mark.usefixtures("start_torchserve")
def test_iris_example(tmpdir):
    iris_dir = os.path.join("examples", "IrisClassification")
    home_dir = os.getcwd()
    run = mlflow.start_run()
    example_command = ["python", os.path.join(iris_dir, "iris_classification.py")]
    extra_files = "{},{}".format(
        os.path.join(iris_dir, "index_to_name.json"),
        os.path.join(home_dir, "model/MLmodel"),
    )
    process.exec_cmd(example_command, cwd=home_dir, env={"MLFLOW_RUN_ID": run.info.run_id})
    create_deployment_command = [
        "python",
        os.path.join(iris_dir, "create_deployment.py"),
        "--export_path",
        os.path.join(home_dir, "model_store"),
        "--serialized_file_path",
        "runs:/{}/model".format(run.info.run_id),
        "--handler",
        os.path.join(iris_dir, "iris_handler.py"),
        "--model_file",
        os.path.join(iris_dir, "iris_classification.py"),
        "--extra_files",
        extra_files,
    ]

    process.exec_cmd(create_deployment_command, cwd=home_dir)
    mlflow.end_run()
    assert os.path.exists(os.path.join(home_dir, "model_store", "iris_classification.mar"))
    predict_command = [
        "python",
        os.path.join(iris_dir, "predict.py"),
        "--input_file_path",
        os.path.join(iris_dir, "sample.json"),
    ]
    res = process.exec_cmd(predict_command, cwd=home_dir)
    assert "SETOSA" in res[1]
