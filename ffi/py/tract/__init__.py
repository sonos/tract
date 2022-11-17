from tract.tract import lib, ffi
import ctypes

class TractError(Exception):
    pass

def nnef():
    return Nnef()

def check(err):
    if err != 0:
        raise TractError(ffi.string(lib.tract_get_last_error()))


class Nnef:
    def __init__(self):
        ptr = ffi.new("TractNnef**")
        check(lib.tract_nnef_create(ptr))
        self.ptr = ptr

    def __del__(self):
        check(lib.tract_nnef_destroy(self.ptr))

    def model_for_path(self, path):
        model = ffi.new("TractModel**")
        path = path.encode("utf-8")
        check(lib.tract_nnef_model_for_path(self.ptr[0], path, model))
        return Model(model)

class Model:
    def __init__(self, ptr):
        self.ptr = ptr

    def __del__(self):
        if self.ptr != ffi.NULL:
            check(lib.tract_model_destroy(self.ptr))

    def optimize(self):
        if self.ptr == ffi.NULL:
            raise TractError("invalid model (maybe already consumed ?)")
        check(lib.tract_model_optimize(self.ptr[0]))

    def into_runnable(self):
        if self.ptr == ffi.NULL:
            raise TractErro("invalid model (maybe already consumed ?)")
        runnable = ffi.new("TractRunnable**")
        check(lib.tract_model_into_runnable(self.ptr, runnable))
        self.ptr = ffi.NULL
        return Runnable(runnable)

class Runnable:
    def __init__(self, ptr):
        self.ptr = ptr

    def __del__(self):
        check(lib.tract_runnable_release(self.ptr))

    def run(self, inputs):
        tensors = [ t.__dlpack__() for t in inputs ]

        print(tensors[0])

        ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
        ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
        ptr = ctypes.pythonapi.PyCapsule_GetPointer(tensors[0], b"dltensor\0")
        ptr = ffi.cast("DLTensor*", ptr);
        print(ptr)

        output = ffi.new("DLTensor**")
        print(output)

        check(lib.tract_runnable_run(self.ptr[0], 1, ptr, 1, output[0]))
        if err != 0:
            raise ffi.string(lib.tract_get_last_error())
