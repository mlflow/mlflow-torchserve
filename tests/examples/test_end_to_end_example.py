import mlflow
import pytest
import os
from mlflow.utils import process


@pytest.mark.usefixtures("start_torchserve")
def test_mnist_example():
    os.environ["MKL_THREADING_LAYER"] = "GNU"
    home_dir = os.getcwd()
    mnist_dir = "examples/MNIST"
    example_command = ["python", "mnist_model.py", "--max_epochs", "1", "--register", "false"]
    process.exec_cmd(example_command, cwd=mnist_dir)

    assert os.path.exists(os.path.join(mnist_dir, "model.pth"))
    create_deployment_command = [
        "python",
        "create_deployment.py",
        "--export_path",
        os.path.join(home_dir, "model_store"),
        "--registered_model_uri",
        "model.pth",
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
    example_command = ["python", os.path.join(iris_dir, "iris_classification.py")]
    extra_files = "{},{}".format(
        os.path.join(iris_dir, "index_to_name.json"),
        os.path.join(home_dir, "model/MLmodel"),
    )
    process.exec_cmd(example_command, cwd=home_dir)
    create_deployment_command = [
        "python",
        os.path.join(iris_dir, "create_deployment.py"),
        "--export_path",
        os.path.join(home_dir, "model_store"),
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
