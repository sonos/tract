from ctypes import *
from typing import Dict, List, Union
from .bindings import check, lib
from .fact import InferenceFact
from .model import Model

class InferenceModel:
    def __init__(self, ptr):
        self.ptr = ptr

    def __del__(self):
        if self.ptr:
            check(lib.tract_inference_model_destroy(byref(self.ptr)))

    def _valid(self):
        if self.ptr == None:
            raise TractError("invalid inference model (maybe already consumed ?)")

    def into_optimized(self) -> Model:
        self._valid()
        model = c_void_p()
        check(lib.tract_inference_model_into_optimized(byref(self.ptr), byref(model)))
        return Model(model)

    def into_typed(self) -> Model:
        self._valid()
        model = c_void_p()
        check(lib.tract_inference_model_into_typed(byref(self.ptr), byref(model)))
        return Model(model)

    def input_count(self) -> int:
        self._valid()
        i = c_size_t()
        check(lib.tract_inference_model_nbio(self.ptr, byref(i), None))
        return i.value

    def output_count(self) -> int:
        self._valid()
        i = c_size_t()
        check(lib.tract_inference_model_nbio(self.ptr, None, byref(i)))
        return i.value

    def input_name(self, input_id: int) -> str:
        self._valid()
        cstring = c_char_p()
        check(lib.tract_inference_model_input_name(self.ptr, input_id, byref(cstring)))
        result = str(cstring.value, "utf-8")
        lib.tract_free_cstring(cstring)
        return result

    def input_fact(self, input_id: int) -> InferenceFact:
        self._valid()
        fact = c_void_p()
        check(lib.tract_inference_model_input_fact(self.ptr, input_id, byref(fact)))
        return InferenceFact(fact)

    def set_input_fact(self, input_id: int, fact: Union[InferenceFact, str, None]) -> None:
        self._valid()
        if isinstance(fact, str):
            fact = self.fact(fact)
        if fact == None:
            check(lib.tract_inference_model_set_input_fact(self.ptr, input_id, None))
        else:
            check(lib.tract_inference_model_set_input_fact(self.ptr, input_id, fact.ptr))

    def output_name(self, output_id: int) -> str:
        self._valid()
        cstring = c_char_p()
        check(lib.tract_inference_model_output_name(self.ptr, output_id, byref(cstring)))
        result = str(cstring.value, "utf-8")
        lib.tract_free_cstring(cstring)
        return result

    def output_fact(self, output_id: int) -> InferenceFact:
        self._valid()
        fact = c_void_p()
        check(lib.tract_inference_model_output_fact(self.ptr, output_id, byref(fact)))
        return InferenceFact(fact)

    def set_output_fact(self, output_id: int, fact: Union[InferenceFact, str, None]) -> None:
        self._valid()
        if isinstance(fact, str):
            fact = self.fact(fact)
        if fact == None:
            check(lib.tract_inference_model_set_output_fact(self.ptr, output_id, None))
        else:
            check(lib.tract_inference_model_set_output_fact(self.ptr, output_id, fact.ptr))

    def fact(self, spec:str) -> InferenceFact:
        self._valid()
        spec = str(spec).encode("utf-8")
        fact = c_void_p();
        check(lib.tract_inference_fact_parse(self.ptr, spec, byref(fact)))
        return InferenceFact(fact)

    def analyse(self) -> None:
        self._valid()
        check(lib.tract_inference_model_analyse(self.ptr, False))

    def into_analysed(self) -> "InferenceModel":
        self.analyse()
        return self
