import json
import numpy
from ctypes import *
from pathlib import Path
from typing import Dict, List, Optional, Union
from .bindings import TractError, check, lib
from .inference_model import InferenceModel
from .tensor import DatumType, dt_numpy_to_tract

class OnnxOptions:
    """
    Options for the ONNX loader, passed to :meth:`Onnx.load`.

    Mirrors ``OnnxOptions`` in the Rust API and serializes to the same JSON.

    .. code-block:: python

        # turn the downsample_ratio input into a constant, so the shapes it
        # drives can be inferred
        options = tract.OnnxOptions().constant_input(
            "downsample_ratio", numpy.array([0.5], dtype=numpy.float32)
        )
        model = tract.onnx().load("./rvm.onnx", options)
    """

    def __init__(self, *, ignore_value_info: bool = False, assertions: Optional[List[str]] = None):
        self._ignore_value_info = ignore_value_info
        self._assertions = list(assertions) if assertions else []
        self._constant_inputs: Dict[str, dict] = {}

    def ignore_value_info(self, ignore: bool = True) -> "OnnxOptions":
        """Discard the shapes the model file declares and let tract infer them. Returns self for chaining."""
        self._ignore_value_info = ignore
        return self

    def assertion(self, assertion: str) -> "OnnxOptions":
        """Add one assertion over the model symbols, e.g. ``"h>=0"``. Returns self for chaining."""
        self._assertions.append(assertion)
        return self

    def constant_input(self, name: str, value: numpy.ndarray) -> "OnnxOptions":
        """
        Replace the input called ``name`` by a constant holding ``value``. Returns self for chaining.

        The input stops being a model input, so analysis can resolve shapes that depend on its
        value rather than only its shape. ``value`` must hold real numbers or booleans.
        """
        value = numpy.asarray(value)
        if value.dtype.kind not in "biuf":
            raise TractError(f"Unsupported dtype for a constant input: {value.dtype}")
        self._constant_inputs[name] = {
            "datum_type": DatumType(dt_numpy_to_tract(value.dtype)).name.lower(),
            "shape": list(value.shape),
            "values": value.astype(numpy.float64).ravel().tolist(),
        }
        return self

    def to_json(self) -> str:
        """Serialize to the JSON the loader expects."""
        return json.dumps({
            "ignore_value_info": self._ignore_value_info,
            "assertions": self._assertions,
            "constant_inputs": self._constant_inputs,
        })

class Onnx:
    """
    Represent the ONNX context in tract.

    It essentially allows to load ONNX models. Note that an ONNX model is loaded as an
    ``InferenceModel`` and not as a ``Model``: many ONNX models come with partial shape and
    element type information, while tract's ``Model`` assume full shape and element type
    knownledge. In this case, it is generally sufficient to inform tract about the input
    shape and type, then let tract *infer* the rest of the missing shape information
    before converting the ``InferenceModel`` to a regular ``Model``.

    .. code-block:: python

        # load the model as an InferenceModel
        model = tract.onnx().load("./mobilenetv2-7.onnx")

        # set the shape and type of its first and only input
        model.set_input_fact(0, "1,3,224,224,f32")

        # get ready to run the model
        model = model.into_model().into_runnable()
    """

    def __init__(self):
        ptr = c_void_p()
        check(lib.tract_onnx_create(byref(ptr)))
        self.ptr = ptr

    def __del__(self):
        check(lib.tract_onnx_destroy(byref(self.ptr)))

    def load(
        self, path: Union[str, Path], options: Union[OnnxOptions, str, None] = None
    ) -> InferenceModel:
        """
        Load an ONNX file as an InferenceModel

        ``options`` is an :class:`OnnxOptions` or the JSON for one, and configures the loader.
        """
        model = c_void_p()
        path = str(path).encode("utf-8")
        if options is None:
            check(lib.tract_onnx_load(self.ptr, path, byref(model)))
        else:
            if isinstance(options, OnnxOptions):
                options = options.to_json()
            check(lib.tract_onnx_load_with_options(self.ptr, path, options.encode("utf-8"), byref(model)))
        return InferenceModel(model)
