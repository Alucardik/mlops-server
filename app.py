import json
from os import getenv

from flask import Flask, request

from models import ModelsRegistry
from model_params import assemble_model_input


models_registry = ModelsRegistry()
models_registry.register_models(log_in_mlflow=False)

first_model_name = models_registry.get_registered_model_names()[0]
inputs = models_registry.get_model_inputs(first_model_name)
label_name = models_registry.get_model_outputs(first_model_name)[0].name

app = Flask(__name__)


@app.route("/predict/<string:model_name>", methods=["PUT"])
def predict(model_name: str):
    body: dict = request.get_json()

    if not models_registry.is_model_registered(model_name):
        return "Unknown model", 404

    try:
        model_input = assemble_model_input(body, inputs)
    except Exception as e:
        print(e)
        return "Invalid input", 400

    return str(models_registry.run_inference(model_name, [label_name], model_input)[0][0][0])


if __name__ == "__main__":
    app.run(host=getenv("SERVER_HOST") or "0.0.0.0", port=int(getenv("SERVER_PORT") or 8080))
