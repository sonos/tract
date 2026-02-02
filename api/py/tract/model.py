import numpy
from ctypes import *
from typing import Dict, List, Union
from .bindings import TractError, check, lib
from .fact import Fact
from .value import Value
from .runnable import Runnable

class Model:
    """
    # Main model object

    ## Central focus point of the model transformation pipeline
 
    The Model is the central point of tract model loading and "model cooking". ONNX and NNEF 
    serialized models are converted to Model (more or less directly) before we can do anything
    of value with them. Model can be dumped to NNEF (or tract-opl which is NNEF plus tract
    proprietary extensions).
    
    A Model can be `optimize()`, substituing the "high level" operators in tract-core operator set by
    the best implementation available for the current system. From there it can be transformed into a 
    Runnable object that we will use to run.
 
    ## Model cooking
 
    But some model transformations can be performed on the Model class:
 
    - declutter (getting rid of training artefacts)
    - "pulsification" (transforming a batch-oriented model into a streaming model)
    - symbol substitution (make N or Batch a fixed number, unlocking potential optimisation later on)
    - static cost evalation and dynamic profiling
    - ...
 
    In some situation, these operation are done "on-the-fly" when a ONNX or NNEF model is loaded,
    at start-up time. In other situation, when start-up time becomes an issue, it may be beneficial
    to "pre-cook" the model: apply the transformations one time, serialize the model as NNEF (with
    tract-opl extension if needed). At start-up this model can be significantly less expensive to
    "cook" for inference.
 
    ## Model and TypedModel

    This class is actually a wrapper around the "TypedModel" in Rust codebase. The "Typed"
    bit means than all shapes and element types in all input, output and temporary values must
    known. There is support in tract for symbols in dimensions, with some limited computation
    capabilities on symbolic expression. For instance, it is relatively frequent to work with
    a Model where all tensors shapes start with the `N` or `Batch`.
    """

    def __init__(self, ptr):
        self.ptr = ptr

    def __del__(self):
        if self.ptr:
            check(lib.tract_model_destroy(byref(self.ptr)))

    def _valid(self):
        if self.ptr == None:
            raise TractError("invalid model (maybe already consumed ?)")

    def input_count(self) -> int:
        """Return the number of inputs of the model"""
        self._valid()
        i = c_size_t()
        check(lib.tract_model_input_count(self.ptr, byref(i)))
        return i.value

    def output_count(self) -> int:
        """Return the number of outputs of the model"""
        self._valid()
        i = c_size_t()
        check(lib.tract_model_output_count(self.ptr, byref(i)))
        return i.value

    def input_name(self, input_id: int) -> str:
        """Return the name of the input_id-th input"""
        self._valid()
        cstring = c_char_p()
        check(lib.tract_model_input_name(self.ptr, input_id, byref(cstring)))
        result = str(cstring.value, "utf-8")
        lib.tract_free_cstring(cstring)
        return result

    def input_fact(self, input_id: int) -> Fact:
        """Return the fact of the input_id-th input"""
        self._valid()
        fact = c_void_p()
        check(lib.tract_model_input_fact(self.ptr, input_id, byref(fact)))
        return Fact(fact)

    def set_output_names(self, names: List[str]):
        """Change the output nodes of the model"""
        self._valid()
        nb = len(names)
        names_str = []
        names_ptr = (c_char_p * nb)()
        for ix, n in enumerate(names):
            names_str.append(str(n).encode("utf-8"))
            names_ptr[ix] = names_str[ix]
        check(lib.tract_model_set_output_names(self.ptr, nb, names_ptr))

    def output_name(self, output_id: int) -> str:
        """Return the name of the output_id-th output"""
        self._valid()
        cstring = c_char_p()
        check(lib.tract_model_output_name(self.ptr, output_id, byref(cstring)))
        result = str(cstring.value, "utf-8")
        lib.tract_free_cstring(cstring)
        return result

    def output_fact(self, input_id: int) -> Fact:
        """Return the fact of the output_id-th output"""
        self._valid()
        fact = c_void_p()
        check(lib.tract_model_output_fact(self.ptr, input_id, byref(fact)))
        return Fact(fact)

    def concretize_symbols(self, values: Dict[str, int]) -> None:
        """Substitute symbols by a value

        Replace all occurencies of the symbols in the dictionary, in all the Model facts shapes.

        While this is not strictly necesary, the optimizing steps may make better choices if the model
        is informed of some specific symbol values.
        """
        self._valid()
        nb = len(values)
        names_str = []
        names = (c_char_p * nb)()
        values_list = (c_int64 * nb)()
        for ix, (k, v) in enumerate(values.items()):
            names_str.append(str(k).encode("utf-8"))
            names[ix] = names_str[ix]
            values_list[ix] = v
        check(lib.tract_model_concretize_symbols(self.ptr, c_size_t(nb), names, values_list))

    def pulse(self, symbol: str, pulse: Union[str, int]) -> None:
        """Pulsify a model.

        `pulse` is typically a one-length dictionary mapping the time dimension symbol to a pulse len.
        """
        self._valid()
        check(lib.tract_model_pulse_simple(byref(self.ptr), symbol.encode("utf-8"), str(pulse).encode("utf-8")))

    def transform(self, transform: str) -> None:
        """Apply a transform to the model
        """
        self._valid()
        check(lib.tract_model_transform(self.ptr, str(transform).encode("utf-8")))

    def into_runnable(self) -> Runnable:
        """Transform the model into a ready to be used Runnable model"""
        self._valid()
        runnable = c_void_p()
        check(lib.tract_model_into_runnable(byref(self.ptr), byref(runnable)))
        return Runnable(runnable)

    def parse_fact(self, spec: str) -> Fact:
        self._valid()
        fact = c_void_p()
        check(lib.tract_model_parse_fact(self.ptr, str(spec).encode("utf-8"), byref(fact)))
        return Fact(fact)

    def property_keys(self) -> List[str]:
        """Query the list of properties names of the model."""
        self._valid()
        count = c_size_t()
        check(lib.tract_model_property_count(self.ptr, byref(count)))
        count = count.value
        cstrings = (POINTER(c_char) * count)()
        check(lib.tract_model_property_names(self.ptr, cstrings))
        names = []
        for i in range(0, count):
            names.append(str(cast(cstrings[i], c_char_p).value, "utf-8"))
            lib.tract_free_cstring(cstrings[i])
        return names

    def property(self, name: str) -> Value:
        """Query a property by name"""
        self._valid()
        value = c_void_p()
        check(lib.tract_model_property(self.ptr, str(name).encode("utf-8"), byref(value)))
        return Value(value)


    def input_facts(self) -> List[Fact]:
        return [ self.input_fact(ix) for ix in range(self.input_count()) ]

    def output_facts(self):
        return [ self.output_fact(ix) for ix in range(self.output_count()) ]
