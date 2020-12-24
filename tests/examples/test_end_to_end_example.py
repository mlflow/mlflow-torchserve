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

    assert os.path.exists(os.path.join(mnist_dir, "model.pth"))
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
    iris_dir_absolute_path = os.path.join(os.getcwd(), iris_dir)
    example_command = ["python", "iris_classification.py"]
    process.exec_cmd(example_command, cwd=iris_dir)
    model_uri = os.path.join(iris_dir_absolute_path, "iris.pt")
    model_file_path = os.path.join(iris_dir_absolute_path, "iris_classification.py")
    handler_file_path = os.path.join(iris_dir_absolute_path, "iris_handler.py")
    extra_file_path = os.path.join(iris_dir_absolute_path, "index_to_name.json")
    input_file_path = os.path.join(iris_dir_absolute_path, "sample.json")
    output_file_path = os.path.join(str(tmpdir), "output.json")
    create_deployment_command = [
        "mlflow deployments create "
        "--name iris_test_29 "
        "--target torchserve "
        "--model-uri {model_uri} "
        '-C "MODEL_FILE={model_file}" '
        '-C "HANDLER={handler_file}" '
        '-C "EXTRA_FILES={extra_file}"'.format(
            model_uri=model_uri,
            model_file=model_file_path,
            handler_file=handler_file_path,
            extra_file=extra_file_path,
        ),
    ]

    process.exec_cmd(create_deployment_command, shell=True)

    predict_command = [
        "mlflow deployments predict "
        "--name iris_test_29 "
        "--target torchserve "
        "--input-path {} --output-path {}".format(input_file_path, output_file_path)
    ]

    process.exec_cmd(predict_command, cwd=iris_dir, shell=True)
    assert os.path.exists(output_file_path)

    with open(output_file_path) as fp:
        result = fp.read()

    assert "SETOSA" in result
