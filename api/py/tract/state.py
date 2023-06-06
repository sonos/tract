import numpy
from ctypes import *
from typing import Dict, List, Union
from .bindings import check, lib
from .fact import Fact
from .value import Value

class State:
    """
    The state of a stateful model.
    """
    def __init__(self, ptr):
        self.ptr = ptr

    def __del__(self):
        check(lib.tract_state_destroy(byref(self.ptr)))

    def _valid(self):
        if self.ptr == None:
            raise TractError("invalid state (maybe already destroyed ?)")

    def input_count(self) -> int:
        """Return the number of inputs of the underlying model"""
        self._valid()
        i = c_size_t()
        check(lib.tract_state_input_count(self.ptr, byref(i)))
        return i.value

    def output_count(self) -> int:
        """Return the number of outputs of the underlying model"""
        self._valid()
        i = c_size_t()
        check(lib.tract_state_output_count(self.ptr, byref(i)))
        return i.value

    def run(self, inputs: List[Union[Value, numpy.ndarray]]) -> List[Value]:
        """
        Runs the model over the provided input list, and returns the model outputs.
        """
        self._valid()
        input_values = []
        for v in inputs:
            if isinstance(v, Value):
                input_values.append(v)
            elif isinstance(v, numpy.ndarray):
                input_values.append(Value.from_numpy(v))
            else:
                raise TractError(f"Inputs must be of type tract.Value or numpy.Array, got {v}")
        input_ptrs = (c_void_p * self.input_count())()
        output_ptrs = (c_void_p * self.output_count())()
        for ix, v in enumerate(input_values):
            input_ptrs[ix] = v.ptr
        check(lib.tract_state_run(self.ptr, input_ptrs, output_ptrs))
        result = []
        for v in output_ptrs:
            result.append(Value(c_void_p(v)))
        return result

    def freeze(self) -> "FrozenState":
        self._valid()
        frozen = c_void_p()
        check(lib.tract_state_freeze(self.ptr, byref(frozen)))
        return FrozenState(frozen)

class FrozenState:
    """
    The state of a stateful model.
    """
    def __init__(self, ptr):
        self.ptr = ptr

    def __del__(self):
        check(lib.tract_frozen_state_destroy(byref(self.ptr)))

    def _valid(self):
        if self.ptr == None:
            raise TractError("invalid frozen state (maybe already destroyed ?)")

    def unfreeze(self) -> State:
        self._valid()
        state = c_void_p()
        check(lib.tract_frozen_state_unfreeze(self.ptr, byref(state)))
        return State(state)
