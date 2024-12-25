from typing import Dict

import onnxruntime as rt
from flask import Flask, request

import model_params as mp


inference_session_v1 = rt.InferenceSession("model_v1.onnx", providers=["CPUExecutionProvider"])
inference_session_v2 = rt.InferenceSession("model_v2.onnx", providers=["CPUExecutionProvider"])
inputs = inference_session_v2.get_inputs()
label_name = inference_session_v2.get_outputs()[0].name
app = Flask(__name__)


def assemble_model_input(body: Dict) -> Dict:
    selected_amenities = set(body.get("amenities", []))
    default_true = {"availability_30", "availability_60", "availability_90", "availability_365", "host_is_superhost"}
    default_85 = {"number_of_reviews_ltm", "review_scores_rating"}
    default_custom = {
        "latitude": 41.40889,
        "longitude": 2.18555,
    }

    model_input = {}
    for inp in inputs:
        input_name = inp.name
        if input_name in default_custom:
            model_input[input_name] = mp.wrap_num_param(default_custom[input_name])
            continue
        if input_name in default_true:
            model_input[input_name] = mp.wrap_num_param(1.)
            continue
        if input_name in default_85:
            model_input[input_name] = mp.wrap_num_param(85.)
            continue
        if input_name.startswith("has_"):
            model_input[input_name] = mp.wrap_num_param(1 if input_name[4:] in selected_amenities else 0)
            continue
        if input_name in mp.num_cols:
            model_input[input_name] = mp.wrap_num_param(body.get(input_name))
            continue
        if input_name in mp.cat_cols:
            model_input[input_name] = mp.wrap_cat_param(body.get(input_name))
            continue

    return model_input


@app.route("/predict/v1", methods=["PUT"])
def predict_v1():
    body: Dict = request.get_json()

    try:
        model_input = assemble_model_input(body)
    except Exception as e:
        print(e)
        return "Invalid input", 400


    return str(inference_session_v1.run([label_name], model_input)[0][0][0])


@app.route("/predict/v2", methods=["PUT"])
def predict_v2():
    body: Dict = request.get_json()

    try:
        model_input = assemble_model_input(body)
    except Exception as e:
        print(e)
        return "Invalid input", 400


    return str(inference_session_v2.run([label_name], model_input)[0][0][0])


if __name__ == "__main__":
    app.run()
