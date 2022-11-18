import numpy
from ctypes import *

lib = CDLL("tract/tract.cpython-39-x86_64-linux-gnu.so")
lib.tract_version.restype = c_char_p
lib.tract_get_last_error.restype = c_char_p

TRACT_DATUM_TYPE_BOOL = 0x01
TRACT_DATUM_TYPE_U8 = 0x11
TRACT_DATUM_TYPE_U16 = 0x12
TRACT_DATUM_TYPE_U32 = 0x13
TRACT_DATUM_TYPE_U64 = 0x14
TRACT_DATUM_TYPE_I8 = 0x21
TRACT_DATUM_TYPE_I16 = 0x22
TRACT_DATUM_TYPE_I32 = 0x23
TRACT_DATUM_TYPE_I64 = 0x24
TRACT_DATUM_TYPE_F16 = 0x32
TRACT_DATUM_TYPE_F32 = 0x33
TRACT_DATUM_TYPE_F64 = 0x34
TRACT_DATUM_TYPE_COMPLEX_I16 = 0x42
TRACT_DATUM_TYPE_COMPLEX_I32 = 0x43
TRACT_DATUM_TYPE_COMPLEX_I64 = 0x44
TRACT_DATUM_TYPE_COMPLEX_F16 = 0x52
TRACT_DATUM_TYPE_COMPLEX_F32 = 0x53
TRACT_DATUM_TYPE_COMPLEX_F64 = 0x54

class TractError(Exception):
    pass

def nnef():
    return Nnef()

def onnx():
    return Onnx()

def check(err):
    if err != 0:
        raise TractError(str(lib.tract_get_last_error()))

class Nnef:
    def __init__(self):
        ptr = c_void_p()
        check(lib.tract_nnef_create(byref(ptr)))
        self.ptr = ptr

    def __del__(self):
        check(lib.tract_nnef_destroy(byref(self.ptr)))

    def model_for_path(self, path):
        model = c_void_p()
        path = path.encode("utf-8")
        check(lib.tract_nnef_model_for_path(self.ptr, path, byref(model)))
        return Model(model)

class Onnx:
    def __init__(self):
        ptr = c_void_p()
        check(lib.tract_onnx_create(byref(ptr)))
        self.ptr = ptr

    def __del__(self):
        check(lib.tract_onnx_destroy(byref(self.ptr)))

    def model_for_path(self, path):
        model = c_void_p()
        path = path.encode("utf-8")
        check(lib.tract_onnx_model_for_path(self.ptr, path, byref(model)))
        return InferenceModel(model)

class InferenceModel:
    def __init__(self, ptr):
        self.ptr = ptr

    def __del__(self):
        if self.ptr:
            check(lib.tract_inference_model_destroy(byref(self.ptr)))

    def into_optimized(self):
        if self.ptr == None:
            raise TractError("invalid inference model (maybe already consumed ?)")
        model = c_void_p()
        check(lib.tract_inference_model_into_optimized(byref(self.ptr), byref(model)))
        return Model(model)

class Model:
    def __init__(self, ptr):
        self.ptr = ptr

    def __del__(self):
        if self.ptr:
            check(lib.tract_model_destroy(byref(self.ptr)))

    def optimize(self):
        if self.ptr == None:
            raise TractError("invalid model (maybe already consumed ?)")
        check(lib.tract_model_optimize(self.ptr))

    def into_runnable(self):
        if self.ptr == None:
            raise TractError("invalid model (maybe already consumed ?)")
        runnable = c_void_p()
        check(lib.tract_model_into_runnable(byref(self.ptr), byref(runnable)))
        return Runnable(runnable)

class Runnable:
    def __init__(self, ptr):
        self.ptr = ptr
        i = c_size_t()
        o = c_size_t()
        check(lib.tract_runnable_nbio(self.ptr, byref(i), byref(o)))
        self.inputs = i.value
        self.outputs = o.value

    def __del__(self):
        check(lib.tract_runnable_release(byref(self.ptr)))

    def run(self, inputs):
        input_ptrs = (c_void_p * self.inputs)()
        output_ptrs = (c_void_p * self.outputs)()
        for ix, v in enumerate(inputs):
            input_ptrs[ix] = v.ptr
        check(lib.tract_runnable_run(self.ptr, input_ptrs, output_ptrs))
        result = []
        for v in output_ptrs:
            result.append(Value(c_void_p(v)))
        return result

class Value:
    def __init__(self, ptr):
        self.ptr = ptr

    def __del__(self):
        if self.ptr:
            check(lib.tract_value_destroy(byref(self.ptr)))

    def from_numpy(array):
        array = numpy.ascontiguousarray(array)

        data = array.__array_interface__['data'][0]
        data = c_void_p(data)
        ptr = c_void_p()
        shape_t = c_size_t * array.ndim
        shape = shape_t()
        for ix in range(0, array.ndim):
            shape[ix] = array.shape[ix]
        check(lib.tract_value_create(TRACT_DATUM_TYPE_F32, c_size_t(array.ndim), shape, data, byref(ptr)))
        return Value(ptr)

    def to_numpy(self):
        if not self.ptr:
            raise TractError("invalid value (already consumed)")
        rank = c_size_t();
        shape = POINTER(c_size_t)()
        dt = c_float
        data = POINTER(dt)()
        check(lib.tract_value_inspect(self.ptr, None, byref(rank), byref(shape), byref(data)))
        rank = rank.value
        shape = [ int(shape[ix]) for ix in range(0, rank) ]
        array = numpy.ctypeslib.as_array(data, shape).copy()
        return array

    def into_numpy(self):
        result = self.to_numpy()
        check(lib.tract_value_destroy(byref(self.ptr)))
        return result

