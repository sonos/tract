from ctypes import *
from pathlib import Path
from typing import Dict, List, Union
from .bindings import check, lib
from .inference_model import InferenceModel

class Onnx:
    def __init__(self):
        ptr = c_void_p()
        check(lib.tract_onnx_create(byref(ptr)))
        self.ptr = ptr

    def __del__(self):
        check(lib.tract_onnx_destroy(byref(self.ptr)))

    def model_for_path(self, path: Union[str, Path]) -> InferenceModel:
        model = c_void_p()
        path = str(path).encode("utf-8")
        check(lib.tract_onnx_model_for_path(self.ptr, path, byref(model)))
        return InferenceModel(model)
