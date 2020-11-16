import sys
import mlflow

if len(sys.argv) != 3:
    raise Exception("python register.py <model_uri> <model_name>")

model_uri = sys.argv[1]
model_name = sys.argv[2]

registered_model = mlflow.register_model(model_uri, model_name)
print("Model Registerd Successfully")
print("Registered Name: ", registered_model.name)
print("Registered Version: ", registered_model.version)