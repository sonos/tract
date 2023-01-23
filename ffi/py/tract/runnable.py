from ctypes import *
from typing import Dict, List, Union # after ctypes so that Union is overriden
import numpy
from .value import Value
from .bindings import check, lib

class Runnable:
    def __init__(self, ptr):
        self.ptr = ptr
        i = c_size_t()
        o = c_size_t()
        check(lib.tract_runnable_nbio(self.ptr, byref(i), byref(o)))
        self.inputs = i.value
        self.outputs = o.value

    def __del__(self):
        check(lib.tract_runnable_release(byref(self.ptr)))

    def _valid(self):
        if self.ptr == None:
            raise TractError("invalid runnable (maybe already consumed ?)")

    def run(self, inputs: List[Union[Value, numpy.ndarray]]) -> List[Value]:
        self._valid()
        input_values = []
        for v in inputs:
            if isinstance(v, Value):
                input_values.append(v)
            elif isinstance(v, numpy.ndarray):
                input_values.append(Value.from_numpy(v))
            else:
                raise TractError(f"Inputs must be of type tract.Value or numpy.Array, got {v}")
        input_ptrs = (c_void_p * self.inputs)()
        output_ptrs = (c_void_p * self.outputs)()
        for ix, v in enumerate(input_values):
            input_ptrs[ix] = v.ptr
        check(lib.tract_runnable_run(self.ptr, input_ptrs, output_ptrs))
        result = []
        for v in output_ptrs:
            result.append(Value(c_void_p(v)))
        return result
