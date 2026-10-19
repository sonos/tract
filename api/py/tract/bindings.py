from ctypes import *
from pathlib import Path

class TractError(Exception):
    pass

if len(list(Path(__file__).parent.glob("*.so"))) > 0:
    dylib_path = list(Path(__file__).parent.glob("*.so"))[0]
elif len(list(Path(__file__).parent.glob("*.pyd"))) > 0:
    dylib_path = list(Path(__file__).parent.glob("*.pyd"))[0]
else:
    raise TractError("Can not find dynamic library")

lib = cdll.LoadLibrary(str(dylib_path))

lib.tract_version.restype = c_char_p
lib.tract_get_last_error.restype = c_char_p
lib.tract_free_cstring.restype = None



def unstable_npyz(name):
    """The libtract function `name`, provided it was built with the `unstable-npyz` feature."""
    try:
        return getattr(lib, name)
    except AttributeError:
        raise TractError(f"{name} unavailable: libtract was built without the unstable-npyz feature")

def take_buffer(data, length) -> bytes:
    """Copy a libtract-allocated buffer to python bytes and free it."""
    result = string_at(data, length.value)
    check(unstable_npyz("tract_free_buffer")(data, length))
    return result

def check(err):
    if err != 0:
        raise TractError(str(lib.tract_get_last_error(), "utf-8"))

