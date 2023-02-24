from ctypes import *
from pathlib import Path
from typing import Dict, List, Union
from .bindings import check, lib
from .inference_model import InferenceModel

class Onnx:
    """
    Represent the ONNX context in tract.

    It essentially allows to load ONNX models. Note that an ONNX model is loaded as an
    `InferenceModel` and not as a `Model`: many ONNX models come with partial shape and
    element type information, while tract's `Model` assume full shape and element type
    knownledge. In this case, it is generally sufficient to inform tract about the input
    shape and type, then let tract *infer* the rest of the missing shape information
    before converting the `InferenceModel` to a regular `Model`.

    ```python
    # load the model as an InferenceModel
    model = tract.onnx().model_for_path("./mobilenetv2-7.onnx")

    # set the shape and type of its first and only input
    model.set_input_fact(0, "1,3,224,224,f32")

    # get ready to run the model
    model = model.into_optimized().into_runnable()
    ```
    """

    def __init__(self):
        ptr = c_void_p()
        check(lib.tract_onnx_create(byref(ptr)))
        self.ptr = ptr

    def __del__(self):
        check(lib.tract_onnx_destroy(byref(self.ptr)))

    def model_for_path(self, path: Union[str, Path]) -> InferenceModel:
        """
        Load an ONNX file as an InferenceModel
        """
        model = c_void_p()
        path = str(path).encode("utf-8")
        check(lib.tract_onnx_model_for_path(self.ptr, path, byref(model)))
        return InferenceModel(model)
