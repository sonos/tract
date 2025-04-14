from ctypes import *
from pathlib import Path
from typing import Dict, List, Union
from .bindings import check, lib
from .model import Model

class Nnef:
    """
    Represent a NNEF context in tract.

    NNEF is a neural model interchange format, similar to ONNX but focusing on the needs
    of an inference engine instead of a training framework.

    `tract` can natively load NNEF models. It can also save models it tract internal format
    as `tract-opl` models. `tract-opl` is a set of proprierary extensions to NNEF allowing to
    serializeing most of the models tract can handle. These extension can be activated by the
    `with_*() methods`.
    """

    def __init__(self):
        ptr = c_void_p()
        check(lib.tract_nnef_create(byref(ptr)))
        self.ptr = ptr

    def __del__(self):
        check(lib.tract_nnef_destroy(byref(self.ptr)))

    def _valid(self):
        if self.ptr == None:
            raise TractError("invalid inference model (maybe already consumed ?)")

    def model_for_path(self, path: Union[str, Path]) -> Model:
        """
        Load an NNEF model from the file or folder at `path`

        ```python
        model = (
            tract.nnef()
            .model_for_path("mobilenet_v2_1.0.onnx.nnef.tgz")
            .into_optimized()
            .into_runnable()
        )
        ```
        """
        self._valid()
        model = c_void_p()
        path = str(path).encode("utf-8")
        check(lib.tract_nnef_model_for_path(self.ptr, path, byref(model)))
        return Model(model)

    def transform_model(self, model: Model, transform_spec: str) -> None:
        """
        Apply a transform to a model. Fails if transform_spec has not been registered.
        Transform is in-place
        """
        self._valid()
        model._valid()
        if not isinstance(model, Model):
            raise TractError("Expected a Model, called with " + model);
        transform_spec = transform_spec.encode("utf-8")
        check(lib.tract_nnef_transform_model(self.ptr, model.ptr, transform_spec))

    def with_tract_core(self) -> "Nnef":
        """
        Enable tract-opl extensions to NNEF to covers tract-core operator set
        """
        self._valid()
        check(lib.tract_nnef_enable_tract_core(self.ptr))
        return self

    def with_tract_extra(self) -> "Nnef":
        """
        Enable tract-extra extensions to NNEF.
        """
        self._valid()
        check(lib.tract_nnef_enable_tract_extra(self.ptr))
        return self

    def with_tract_transformers(self) -> "Nnef":
        """
        Enable tract-transformers extensions to NNEF.
        """
        self._valid()
        check(lib.tract_nnef_enable_tract_transformers(self.ptr))
        return self

    def with_onnx(self) -> "Nnef":
        """
        Enable tract-opl extensions to NNEF to covers (more or) ONNX operator set
        """
        self._valid()
        check(lib.tract_nnef_enable_onnx(self.ptr))
        return self

    def with_pulse(self) -> "Nnef":
        """
        Enable tract-opl extensions to NNEF for tract pulse operators (for audio streaming)
        """
        self._valid()
        check(lib.tract_nnef_enable_pulse(self.ptr))
        return self

    def with_extended_identifier_syntax(self) -> "Nnef":
        """
        Enable tract-opl extensions to NNEF for extended identifiers (will support PyTorch 2 path-like ids)
        """
        self._valid()
        check(lib.tract_nnef_enable_extended_identifier_syntax(self.ptr, True))
        return self

    def write_model_to_dir(self, model: Model, path: Union[str, Path]) -> None:
        """
        Save `model` as a NNEF directory model in `path`.

        tract tries to stick to strict NNEF even if extensions has been enabled.
        """
        self._valid()
        model._valid()
        if not isinstance(model, Model):
            raise TractError("Expected a Model, called with " + model);
        path = str(path).encode("utf-8")
        check(lib.tract_nnef_write_model_to_dir(self.ptr, path, model.ptr))

    def write_model_to_tar(self, model: Model, path: Union[str, Path]) -> None:
        """
        Save `model` as a NNEF tar archive in `path`.

        tract tries to stick to strict NNEF even if extensions has been enabled.
        """
        self._valid()
        model._valid()
        if not isinstance(model, Model):
            raise TractError("Expected a Model, called with " + model);
        path = str(path).encode("utf-8")
        check(lib.tract_nnef_write_model_to_tar(self.ptr, path, model.ptr))

    def write_model_to_tar_gz(self, model: Model, path: Union[str, Path]) -> None:
        """
        Save `model` as a NNEF tar compressed archive in `path`.

        tract tries to stick to strict NNEF even if extensions has been enabled.
        """
        self._valid()
        model._valid()
        if not isinstance(model, Model):
            raise TractError("Expected a Model, called with " + model);
        path = str(path).encode("utf-8")
        check(lib.tract_nnef_write_model_to_tar_gz(self.ptr, path, model.ptr))

