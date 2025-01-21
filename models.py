from typing import Optional, Any
from pathlib import Path
from os import getenv

from onnxruntime import InferenceSession, NodeArg


class ModelsRegistry:
    _models_dir: Path = Path(getenv("MODELS_DIR") or "./").resolve()
    _inference_sessions: dict[str, InferenceSession] = {}

    def __init__(self, models_dir: str = None):
        if models_dir and Path(models_dir).is_dir():
            self._models_dir = Path(models_dir).resolve()


    def register_models(self):
        print("registering models from", self._models_dir)
        print("only onnx models are supported")

        for p in self._models_dir.iterdir():
            file_name_parts = p.name.split(".")
            if file_name_parts[-1] != "onnx" or not p.is_file():
                continue

            model_name = "_".join(file_name_parts[:-1])
            if self._inference_sessions.get(model_name) is not None:
                print("encountered model", p.name, "several times, only the first occurrence is registered")
                continue

            self._inference_sessions[model_name] = InferenceSession(p, providers=["CPUExecutionProvider"])
            print("registered model:", model_name)

    def get_registered_model_names(self) -> list[str]:
        return list(self._inference_sessions.keys())

    def is_model_registered(self, model_name: str) -> bool:
        return model_name in self._inference_sessions

    def get_model_inputs(self, model_name: str) -> Optional[list[NodeArg]]:
        if model_name in self._inference_sessions:
            return self._inference_sessions[model_name].get_inputs()

        return None

    def get_model_outputs(self, model_name: str) -> Optional[list[NodeArg]]:
        if model_name in self._inference_sessions:
            return self._inference_sessions[model_name].get_outputs()

        return None

    def run_inference(self, model_name: str, output_names: list[str], inputs: dict[str, Any]) -> Optional[Any]:
        if model_name not in self._inference_sessions:
            return None

        self._inference_sessions[model_name].run(output_names, inputs)
