import pickle
import json
import numpy as np


def model_fn(model_dir):
    with open(f"{model_dir}/diabetes_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model


def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        return np.array(data['inputs'])
    raise ValueError("Unsupported Content Type")


def predict_fn(input_data, model):
    return model.predict(input_data)


def output_fn(prediction, content_type):
    return json.dumps({'predictions': prediction.tolist()})
