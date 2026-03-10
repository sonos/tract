from ctypes import *
from typing import Dict, List, Union # after ctypes so that Union is overriden
import numpy
from .runnable import Runnable
from .model import Model
from .bindings import TractError, check, lib

def runtime_for_name(name: str):
    runtime = c_void_p()
    check(lib.tract_runtime_for_name(str(name).encode("utf-8"), byref(runtime)))
    return Runtime(runtime)

class Runtime:
    """
    Represents a harware/software stack that can execute a Model.

    The main point of interest is the "prepare" method that transform a Model
    into a Runnable for this Runtime.
    """
    def __init__(self, ptr):
        self.ptr = ptr

    def __del__(self):
        check(lib.tract_runtime_release(byref(self.ptr)))

    def _valid(self):
        if self.ptr == None:
            raise TractError("invalid runtime (maybe already consumed ?)")
   
    def name(self) -> str:
        """Return the name of this Runtime."""
        self._valid()
        ptr = c_char_p()
        check(lib.tract_runtime_name(self.ptr, byref(ptr)))
        result = ptr.value.decode("utf-8")
        lib.tract_free_cstring(ptr)
        return result

    def prepare(self, model:Model) -> Runnable:
        """Prepare a model for execution on the Runtime.

        NB: The passed model is invalidated by this call.
        """
        self._valid()
        runnable = c_void_p()
        check(lib.tract_runtime_prepare(self.ptr, byref(model.ptr), byref(runnable)))
        model.ptr = None
        return Runnable(runnable)

    
