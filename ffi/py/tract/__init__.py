import numpy
from ctypes import *
from pathlib import Path
from typing import Dict, List, Union

from .bindings import check, lib, TractError
from .value import Value
from .fact import Fact, InferenceFact
from .model import Model
from .inference_model import InferenceModel
from .runnable import Runnable
from .nnef import Nnef
from .onnx import Onnx

def version() -> str:
    print(lib)
    return str(lib.tract_version(), "utf-8")

def nnef() -> Nnef:
    return Nnef()

def onnx() -> Onnx:
    return Onnx()

