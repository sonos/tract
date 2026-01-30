from ctypes import *
from tract.bindings import TractError, check, lib
from typing import Dict

class Dim:
    "Tract dimension for tensor, supporting symbols"

    def __init__(self, ptr):
        self.ptr = ptr

    def __del__(self):
        if self.ptr:
            check(lib.tract_dim_destroy(byref(self.ptr)))

    def __str__(self):
        return self.dump()

    def __int__(self):
        return self.to_int64()

    def _valid(self):
        if self.ptr == None:
            raise TractError("invalid dim")

    def dump(self):
        self._valid()
        cstring = c_char_p();
        check(lib.tract_dim_dump(self.ptr, byref(cstring)))
        result = str(cstring.value, "utf-8")
        lib.tract_free_cstring(cstring)
        return result

    def eval(self, values: Dict[str, int]) -> "Dim":
        self._valid()
        nb = len(values)
        names_str = []
        names = (c_char_p * nb)()
        values_list = (c_int64 * nb)()
        for ix, (k, v) in enumerate(values.items()):
            names_str.append(str(k).encode("utf-8"))
            names[ix] = names_str[ix]
            values_list[ix] = v
        ptr = c_void_p()
        check(lib.tract_dim_eval(self.ptr, c_size_t(nb), names, values_list, byref(ptr)))
        return Dim(ptr)

    def to_int64(self) -> int:
        self._valid()
        i = c_int64()
        check(lib.tract_dim_to_int64(self.ptr, byref(i)))
        return i.value
