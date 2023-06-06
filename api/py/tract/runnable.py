from ctypes import *
from typing import Dict, List, Union # after ctypes so that Union is overriden
import numpy
from .value import Value
from .state import State
from .bindings import check, lib

class Runnable:
    """
    A model in the Runnable state is ready to perform computation.
    """
    def __init__(self, ptr):
        self.ptr = ptr

    def __del__(self):
        check(lib.tract_runnable_release(byref(self.ptr)))

    def _valid(self):
        if self.ptr == None:
            raise TractError("invalid runnable (maybe already consumed ?)")

    def input_count(self) -> int:
        """Return the number of inputs of the underlying model"""
        self._valid()
        i = c_size_t()
        check(lib.tract_runnable_input_count(self.ptr, byref(i)))
        return i.value

    def output_count(self) -> int:
        """Return the number of outputs of the underlying model"""
        self._valid()
        i = c_size_t()
        check(lib.tract_runnable_output_count(self.ptr, byref(i)))
        return i.value

    def run(self, inputs: List[Union[Value, numpy.ndarray]]) -> List[Value]:
        """
        Runs the model over the provided input list, and returns the model outputs.
        """
        return self.spawn_state().run(inputs)

    def spawn_state(self):
        self._valid()
        state = c_void_p()
        check(lib.tract_runnable_spawn_state(self.ptr, byref(state)))
        return State(state)
