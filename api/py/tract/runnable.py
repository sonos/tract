from ctypes import *
from typing import Dict, List, Union # after ctypes so that Union is overriden
import numpy
from .fact import Fact
from .value import Value
from .state import State
from .bindings import TractError, check, lib

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

    def input_fact(self, input_id: int) -> Fact:
        """Return the fact of the input_id-th input"""
        self._valid()
        fact = c_void_p()
        check(lib.tract_runnable_input_fact(self.ptr, input_id, byref(fact)))
        return Fact(fact)

    def output_fact(self, output_id: int) -> Fact:
        """Return the fact of the output_id-th output"""
        self._valid()
        fact = c_void_p()
        check(lib.tract_runnable_output_fact(self.ptr, output_id, byref(fact)))
        return Fact(fact)

    def property_keys(self) -> List[str]:
        """Query the list of properties names of the runnable model."""
        self._valid()
        count = c_size_t()
        check(lib.tract_runnable_property_count(self.ptr, byref(count)))
        count = count.value
        cstrings = (POINTER(c_char) * count)()
        check(lib.tract_runnable_property_names(self.ptr, cstrings))
        names = []
        for i in range(0, count):
            names.append(str(cast(cstrings[i], c_char_p).value, "utf-8"))
            lib.tract_free_cstring(cstrings[i])
        return names

    def property(self, name: str) -> Value:
        """Query a property by name"""
        self._valid()
        value = c_void_p()
        check(lib.tract_runnable_property(self.ptr, str(name).encode("utf-8"), byref(value)))
        return Value(value)

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

    def profile_json(self, inputs: Union[None, List[Union[Value, numpy.ndarray]]], state_initializers: Union[None, List[Union[Value, numpy.ndarray]]]) -> str:
        """Profile the model. Also compute the static costs of operators.

        Returns is a json buffer.
        """
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

        state_values = []
        state_ptrs = None
        n_states = 0
        if state_initializers != None:
            n_states = len(state_initializers)
            state_ptrs = (c_void_p * n_states)()

            for ix, v in enumerate(state_initializers):
                if isinstance(v, Value):
                    state_values.append(v)
                elif isinstance(v, numpy.ndarray):
                    state_values.append(Value.from_numpy(v))
                else:
                    raise TractError(f"State values must be of type tract.Value or numpy.Array, got {v}")

                state_ptrs[ix] = state_values[ix].ptr

        check(lib.tract_runnable_profile_json(self.ptr, input_ptrs, state_ptrs, n_states, byref(cstring)))
        result = str(cstring.value, "utf-8")
        lib.tract_free_cstring(cstring)
        return result

    def input_facts(self) -> List[Fact]:
        return [ self.input_fact(ix) for ix in range(self.input_count()) ]

    def output_facts(self):
        return [ self.output_fact(ix) for ix in range(self.output_count()) ]
