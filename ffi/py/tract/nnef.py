from ctypes import *
from pathlib import Path
from typing import Dict, List, Union
from .bindings import check, lib
from .model import Model

class Nnef:
    def __init__(self):
        ptr = c_void_p()
        check(lib.tract_nnef_create(byref(ptr)))
        self.ptr = ptr

    def __del__(self):
        check(lib.tract_nnef_destroy(byref(self.ptr)))

    def _valid(self):
        if self.ptr == None:
            raise TractError("invalid inference model (maybe already consumed ?)")

    def valid(self):
        if self.ptr == None:
            raise TractError("invalid inference model (maybe already consumed ?)")

    def model_for_path(self, path: Union[str, Path]) -> Model:
        self._valid()
        model = c_void_p()
        path = str(path).encode("utf-8")
        check(lib.tract_nnef_model_for_path(self.ptr, path, byref(model)))
        return Model(model)

    def with_tract_core(self) -> "Nnef":
        self._valid()
        check(lib.tract_nnef_enable_tract_core(self.ptr))
        return self

    def with_onnx(self) -> "Nnef":
        self._valid()
        check(lib.tract_nnef_enable_onnx(self.ptr))
        return self

    def with_pulse(self) -> "Nnef":
        self._valid()
        check(lib.tract_nnef_enable_pulse(self.ptr))
        return self

    def write_model_to_dir(self, model: Model, path: Union[str, Path]) -> None:
        self._valid()
        model._valid()
        if not isinstance(model, Model):
            raise TractError("Expected a Model, called with " + model);
        path = str(path).encode("utf-8")
        check(lib.tract_nnef_write_model_to_dir(self.ptr, path, model.ptr))

    def write_model_to_tar(self, model: Model, path: Union[str, Path]) -> None:
        self._valid()
        model._valid()
        if not isinstance(model, Model):
            raise TractError("Expected a Model, called with " + model);
        path = str(path).encode("utf-8")
        check(lib.tract_nnef_write_model_to_tar(self.ptr, path, model.ptr))

    def write_model_to_tar_gz(self, model: Model, path: Union[str, Path]) -> None:
        self._valid()
        model._valid()
        if not isinstance(model, Model):
            raise TractError("Expected a Model, called with " + model);
        path = str(path).encode("utf-8")
        check(lib.tract_nnef_write_model_to_tar_gz(self.ptr, path, model.ptr))

