import json

import mlflow.pytorch
import numpy as np
import torch
from iris_classification import IrisClassification


def predict(model):
    model.eval()
    inference_output = model(torch.tensor([4.4000, 3.0000, 1.3000, 0.2000]))
    predicted_idx = str(np.argmax(inference_output.cpu().detach().numpy()))
    with open("index_to_name.json") as fp:
        data = json.load(fp)
        return data[predicted_idx]

# Save as state dict, load the model and predict


iris = IrisClassification()
s_model = mlflow.pytorch.load_state_dict("model_state_dict", iris)
print("State dict prediction Result:", predict(s_model))


# Save the entire model, load the model and predict

e_model = mlflow.pytorch.load_model("entire_model")
print("Entire model prediction Result:", predict(e_model))


