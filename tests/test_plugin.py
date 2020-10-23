from mlflow import deployments


f_model_uri = "fake_model_uri"
f_deployment_id = "fake_deployment_name"
f_flavor = None
f_target = "torchserve"


def test_create_success():
    client = deployments.get_deploy_client(f_target)
    ret = client.create_deployment(f_deployment_id, f_model_uri, f_flavor, config={})
    assert isinstance(ret, dict)
    assert ret["name"] == f_deployment_id
    assert ret["flavor"] == f_flavor
