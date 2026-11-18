import json
from ctypes import *
from typing import Dict, List, Optional, Union # after ctypes so that Union is overriden
import numpy
from .runnable import Runnable
from .model import Model
from .bindings import TractError, check, lib

def runtime_for_name(name: str):
    """Look up a runtime by name and return a ``Runtime`` instance.

    Available runtimes depend on the build and the platform: ``"default"`` for
    CPU, ``"metal"`` on Apple Silicon Macs, ``"cuda"`` on systems with NVIDIA
    GPUs.
    """
    runtime = c_void_p()
    check(lib.tract_runtime_for_name(str(name).encode("utf-8"), byref(runtime)))
    return Runtime(runtime)

class RunOptions:
    """
    Options for :meth:`Runtime.prepare`.

    Mirrors ``RunOptions`` in the Rust API and serializes to the same JSON.

    .. code-block:: python

        # size GPU memory for sequences of 512, without fixing the input shape
        options = tract.RunOptions().memory_sizing_hint("S", 512)
        runnable = tract.runtime_for_name("metal").prepare(model, options)
    """

    def __init__(self, *, memory_sizing_hints: Optional[Dict[str, int]] = None):
        self._memory_sizing_hints = dict(memory_sizing_hints) if memory_sizing_hints else {}

    def memory_sizing_hint(self, name: str, value: int) -> "RunOptions":
        """
        Size memory for symbol ``name`` taking ``value``. Returns self for chaining.

        Hints only steer memory planning: inputs above or below them still run. ``name`` must be a
        symbol of the prepared model and ``value`` nonnegative.
        """
        self._memory_sizing_hints[name] = value
        return self

    def to_json(self) -> str:
        """Serialize to the JSON the runtime expects."""
        return json.dumps({"memory_sizing_hints": self._memory_sizing_hints})

class Runtime:
    """
    A hardware/software backend that can execute a ``Model``.

    Calling ``Model.into_runnable()`` implicitly uses the default CPU runtime.
    To run on a GPU, obtain a ``Runtime`` via :func:`runtime_for_name` —
    ``"metal"`` on Apple Silicon Macs, ``"cuda"`` on NVIDIA systems — then
    call :meth:`prepare` to produce a ``Runnable`` optimized for that backend.
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

    def prepare(self, model:Model, options: Union[RunOptions, str, None] = None) -> Runnable:
        """Prepare a model for execution on the Runtime.

        ``options`` is a :class:`RunOptions` or the JSON for one.

        NB: The passed model is invalidated by this call.
        """
        self._valid()
        runnable = c_void_p()
        if options is None:
            check(lib.tract_runtime_prepare(self.ptr, byref(model.ptr), byref(runnable)))
        else:
            if isinstance(options, RunOptions):
                options = options.to_json()
            check(lib.tract_runtime_prepare_with_options(
                self.ptr, byref(model.ptr), options.encode("utf-8"), byref(runnable)))
        model.ptr = None
        return Runnable(runnable)

    
