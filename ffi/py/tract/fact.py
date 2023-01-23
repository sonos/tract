from ctypes import *
from tract.bindings import check, lib

class InferenceFact:
    def __init__(self, ptr):
        self.ptr = ptr

    def __del__(self):
        if self.ptr:
            check(lib.tract_inference_fact_destroy(byref(self.ptr)))

    def __str__(self):
        return self.dump()

    def _valid(self):
        if self.ptr == None:
            raise TractError("invalid inference fact (maybe already consumed ?)")

    def dump(self):
        self._valid()
        cstring = c_char_p();
        check(lib.tract_inference_fact_dump(self.ptr, byref(cstring)))
        result = str(cstring.value, "utf-8")
        lib.tract_free_cstring(cstring)
        return result

class Fact:
    def __init__(self, ptr):
        self.ptr = ptr

    def __del__(self):
        if self.ptr:
            check(lib.tract_fact_destroy(byref(self.ptr)))

    def __str__(self):
        return self.dump()

    def _valid(self):
        if self.ptr == None:
            raise TractError("invalid fact (maybe already consumed ?)")

    def dump(self):
        self._valid()
        cstring = c_char_p();
        check(lib.tract_fact_dump(self.ptr, byref(cstring)))
        result = str(cstring.value, "utf-8")
        lib.tract_free_cstring(cstring)
        return result

