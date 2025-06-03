import numpy
from ctypes import *
from typing import Dict, List, Union
from .bindings import TractError, check, lib
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
    
    def initializable_states_count(self) -> int:
        """
        Get number of ops with an initializable state
        """
        self._valid()
        n_states = c_size_t()
        check(lib.tract_state_initializable_states_count(self.ptr, byref(n_states)))
        return n_states.value

    def get_states_facts(self) -> List[Fact]:
        """
        Get Stateful Ops' state facts
        """
        self._valid()

        n_states = self.initializable_states_count()
        print(n_states)
        fact_ptrs = (c_void_p * n_states)()

        check(lib.tract_state_get_states_facts(self.ptr, fact_ptrs))

        res = []
        for i in range(n_states):
            res.append(Fact(c_void_p(fact_ptrs[i])))

        return res

    def set_states(self, states: List[Union[Value, numpy.ndarray]]):
        """
        Initialize Stateful Ops with given states
        """
        self._valid()

        n_states = len(states)
        if n_states != self.initializable_states_count():
            raise TractError("Invalid number of entries in given state list")

        state_values = []
        state_ptrs = (c_void_p * n_states)()

        for ix, v in enumerate(states):
            if isinstance(v, Value):
                state_values.append(v)
            elif isinstance(v, numpy.ndarray):
                state_values.append(Value.from_numpy(v))
            else:
                raise TractError(f"State values must be of type tract.Value or numpy.Array, got {v}")

            state_ptrs[ix] = state_values[ix].ptr

        check(lib.tract_state_set_states(self.ptr, state_ptrs))

    def get_states(self) -> List[Value]:
        """
        Get Stateful Ops' current states
        """
        self._valid()

        n_states = self.initializable_states_count()
        state_ptrs = (c_void_p * n_states)()

        check(lib.tract_state_get_states(self.ptr, state_ptrs))

        res = []
        for i in range(n_states):
            res.append(Value(c_void_p(state_ptrs[i])))

        return res

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
