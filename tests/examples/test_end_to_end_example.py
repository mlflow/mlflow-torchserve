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
    example_command = ["python", "iris_classification.py"]
    process.exec_cmd(example_command, cwd=iris_dir)
    create_deployment_command = ["python", "iris_classification.py"]

    process.exec_cmd(create_deployment_command, cwd=iris_dir)

    assert os.path.exists(os.path.join(home_dir, "model_store", "iris_classification.mar"))
    predict_command = ["python", "predict.py"]
    res = process.exec_cmd(predict_command, cwd=iris_dir)
    assert "SETOSA" in res[1]
