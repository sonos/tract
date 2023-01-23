import numpy
from ctypes import *
from typing import Dict, List, Union
from .bindings import check, lib
from .fact import Fact
from .value import Value
from .runnable import Runnable

class Model:
    def __init__(self, ptr):
        self.ptr = ptr

    def __del__(self):
        if self.ptr:
            check(lib.tract_model_destroy(byref(self.ptr)))

    def _valid(self):
        if self.ptr == None:
            raise TractError("invalid model (maybe already consumed ?)")

    def input_count(self) -> int:
        self._valid()
        i = c_size_t()
        check(lib.tract_model_nbio(self.ptr, byref(i), None))
        return i.value

    def output_count(self) -> int:
        self._valid()
        i = c_size_t()
        check(lib.tract_model_nbio(self.ptr, None, byref(i)))
        return i.value

    def input_name(self, input_id: int) -> str:
        self._valid()
        cstring = c_char_p()
        check(lib.tract_model_input_name(self.ptr, input_id, byref(cstring)))
        result = str(cstring.value, "utf-8")
        lib.tract_free_cstring(cstring)
        return result

    def input_fact(self, input_id: int) -> Fact:
        self._valid()
        fact = c_void_p()
        check(lib.tract_model_input_fact(self.ptr, input_id, byref(fact)))
        return Fact(fact)

    def output_name(self, output_id: int) -> str:
        self._valid()
        cstring = c_char_p()
        check(lib.tract_model_output_name(self.ptr, output_id, byref(cstring)))
        result = str(cstring.value, "utf-8")
        lib.tract_free_cstring(cstring)
        return result

    def output_fact(self, input_id: int) -> Fact:
        self._valid()
        fact = c_void_p()
        check(lib.tract_model_output_fact(self.ptr, input_id, byref(fact)))
        return Fact(fact)

    def concretize_symbols(self, values: Dict[str, int]) -> None:
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
        self._valid()
        check(lib.tract_model_pulse_simple(byref(self.ptr), symbol.encode("utf-8"), str(pulse).encode("utf-8")))

    def declutter(self) -> None:
        self._valid()
        check(lib.tract_model_declutter(self.ptr))

    def into_decluttered(self) -> "Model":
        self.declutter();
        return self

    def optimize(self) -> None:
        self._valid()
        check(lib.tract_model_optimize(self.ptr))

    def into_optimized(self) -> "Model":
        self.optimize()
        return self

    def into_runnable(self) -> Runnable:
        self._valid()
        runnable = c_void_p()
        check(lib.tract_model_into_runnable(byref(self.ptr), byref(runnable)))
        return Runnable(runnable)

    def property_keys(self) -> List[str]:
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
        self._valid()
        value = c_void_p()
        check(lib.tract_model_property(self.ptr, str(name).encode("utf-8"), byref(value)))
        return Value(value)

    def profile_json(self, inputs: Union[None, List[Union[Value, numpy.ndarray]]]) -> str:
        self._valid()
        cstring = c_char_p()
        input_values = []
        input_ptrs = None
        if inputs != None:
            for v in inputs:
                if isinstance(v, Value):
                    input_values.append(v)
                elif isinstance(v, numpy.ndarray):
                    input_values.append(Value.from_numpy(v))
                else:
                    raise TractError(f"Inputs must be of type tract.Value or numpy.Array, got {v}")
            input_ptrs = (c_void_p * len(inputs))()
            for ix, v in enumerate(input_values):
                input_ptrs[ix] = v.ptr
        check(lib.tract_model_profile_json(self.ptr, input_ptrs, byref(cstring)))
        result = str(cstring.value, "utf-8")
        lib.tract_free_cstring(cstring)
        return result

