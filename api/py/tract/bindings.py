from ctypes import *
from pathlib import Path

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

class TractError(Exception):
    pass

def check(err):
    if err != 0:
        raise TractError(str(lib.tract_get_last_error(), "utf-8"))

