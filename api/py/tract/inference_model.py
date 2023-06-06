from ctypes import *
from typing import Dict, List, Union
from .bindings import check, lib
from .fact import InferenceFact
from .model import Model

class InferenceModel:
    """
    ONNX model are loaded as an
    `InferenceModel`s instead of `Model`s: many ONNX models come with partial shape and
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
    def __init__(self, ptr):
        self.ptr = ptr

    def __del__(self):
        if self.ptr:
            check(lib.tract_inference_model_destroy(byref(self.ptr)))

    def _valid(self):
        if self.ptr == None:
            raise TractError("invalid inference model (maybe already consumed ?)")

    def into_optimized(self) -> Model:
        """
        Run the InferenceModel through the full tract optimisation pipeline to get an
        optimised Model.
        """
        self._valid()
        model = c_void_p()
        check(lib.tract_inference_model_into_optimized(byref(self.ptr), byref(model)))
        return Model(model)

    def into_typed(self) -> Model:
        """
        Convert an InferenceModel to a regular typed `Model`.

        This will leave the opportunity to run more transformation on the intermediary form of the
        model, before optimisint it all the way.
        """
        self._valid()
        model = c_void_p()
        check(lib.tract_inference_model_into_typed(byref(self.ptr), byref(model)))
        return Model(model)

    def input_count(self) -> int:
        """Return the number of inputs of the model"""
        self._valid()
        i = c_size_t()
        check(lib.tract_inference_model_input_count(self.ptr, byref(i)))
        return i.value

    def output_count(self) -> int:
        """Return the number of outputs of the model"""
        self._valid()
        i = c_size_t()
        check(lib.tract_inference_model_output_count(self.ptr, byref(i)))
        return i.value

    def input_name(self, input_id: int) -> str:
        """Return the name of the `input_id`th input."""
        self._valid()
        cstring = c_char_p()
        check(lib.tract_inference_model_input_name(self.ptr, input_id, byref(cstring)))
        result = str(cstring.value, "utf-8")
        lib.tract_free_cstring(cstring)
        return result

    def input_fact(self, input_id: int) -> InferenceFact:
        """Extract the InferenceFact of the `input_id`th input."""
        self._valid()
        fact = c_void_p()
        check(lib.tract_inference_model_input_fact(self.ptr, input_id, byref(fact)))
        return InferenceFact(fact)

    def set_input_fact(self, input_id: int, fact: Union[InferenceFact, str, None]) -> None:
        """Change the InferenceFact of the `input_id`th input."""
        self._valid()
        if isinstance(fact, str):
            fact = self.fact(fact)
        if fact == None:
            check(lib.tract_inference_model_set_input_fact(self.ptr, input_id, None))
        else:
            check(lib.tract_inference_model_set_input_fact(self.ptr, input_id, fact.ptr))

    def set_output_names(self, names: List[str]):
        """Change the output nodes of the model"""
        self._valid()
        nb = len(names)
        names_str = []
        names_ptr = (c_char_p * nb)()
        for ix, n in enumerate(names):
            names_str.append(str(n).encode("utf-8"))
            names_ptr[ix] = names_str[ix]
        check(lib.tract_inference_model_set_output_names(self.ptr, nb, names_ptr))

    def output_name(self, output_id: int) -> str:
        """Return the name of the `output_id`th output."""
        self._valid()
        cstring = c_char_p()
        check(lib.tract_inference_model_output_name(self.ptr, output_id, byref(cstring)))
        result = str(cstring.value, "utf-8")
        lib.tract_free_cstring(cstring)
        return result

    def output_fact(self, output_id: int) -> InferenceFact:
        """Extract the InferenceFact of the `output_id`th output."""
        self._valid()
        fact = c_void_p()
        check(lib.tract_inference_model_output_fact(self.ptr, output_id, byref(fact)))
        return InferenceFact(fact)

    def set_output_fact(self, output_id: int, fact: Union[InferenceFact, str, None]) -> None:
        """Change the InferenceFact of the `output_id`th output."""
        self._valid()
        if isinstance(fact, str):
            fact = self.fact(fact)
        if fact == None:
            check(lib.tract_inference_model_set_output_fact(self.ptr, output_id, None))
        else:
            check(lib.tract_inference_model_set_output_fact(self.ptr, output_id, fact.ptr))

    def fact(self, spec:str) -> InferenceFact:
        """
        Parse an fact specification as an `InferenceFact`

        Typical `InferenceFact` specification is in the form "1,224,224,3,f32". Comma-separated
        list of dimension, one for each axis, plus an mnemonic for the element type. f32 is 
        single precision "float", i16 is a 16-bit signed integer, and u8 a 8-bit unsigned integer.
        """
        self._valid()
        spec = str(spec).encode("utf-8")
        fact = c_void_p();
        check(lib.tract_inference_fact_parse(self.ptr, spec, byref(fact)))
        return InferenceFact(fact)

    def analyse(self) -> None:
        """
        Perform shape and element type inference on the model.
        """
        self._valid()
        check(lib.tract_inference_model_analyse(self.ptr, False))

    def into_analysed(self) -> "InferenceModel":
        """
        Perform shape and element type inference on the model.
        """
        self.analyse()
        return self
