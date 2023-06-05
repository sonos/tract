import numpy
from ctypes import *
from typing import Dict, List, Union
from tract.bindings import check, lib

TRACT_DATUM_TYPE_BOOL = 0x01
TRACT_DATUM_TYPE_U8 = 0x11
TRACT_DATUM_TYPE_U16 = 0x12
TRACT_DATUM_TYPE_U32 = 0x14
TRACT_DATUM_TYPE_U64 = 0x18
TRACT_DATUM_TYPE_I8 = 0x21
TRACT_DATUM_TYPE_I16 = 0x22
TRACT_DATUM_TYPE_I32 = 0x24
TRACT_DATUM_TYPE_I64 = 0x28
TRACT_DATUM_TYPE_F16 = 0x32
TRACT_DATUM_TYPE_F32 = 0x34
TRACT_DATUM_TYPE_F64 = 0x38
TRACT_DATUM_TYPE_COMPLEX_I16 = 0x42
TRACT_DATUM_TYPE_COMPLEX_I32 = 0x44
TRACT_DATUM_TYPE_COMPLEX_I64 = 0x48
TRACT_DATUM_TYPE_COMPLEX_F16 = 0x52
TRACT_DATUM_TYPE_COMPLEX_F32 = 0x54
TRACT_DATUM_TYPE_COMPLEX_F64 = 0x58

def dt_numpy_to_tract(dt):
    if dt.kind == 'b':
        return TRACT_DATUM_TYPE_BOOL
    if dt.kind == 'u':
        return 0x10 + dt.itemsize
    if dt.kind == 'i':
        return 0x20 + dt.itemsize
    if dt.kind == 'f':
        return 0x30 + dt.itemsize
    if dt.kind == 'c':
        return 0x50 + dt.itemsize / 2
    raise TractError("Unsupported Numpy dtype: " + dt)


class Value:
    """
    Represents a tensor suitable for manipulation by tract.

    On the Python side, the main way to access tensor data is to
    convert the value to a numpy array.
    """
    def __init__(self, ptr):
        self.ptr = ptr

    def __del__(self):
        if self.ptr:
            check(lib.tract_value_destroy(byref(self.ptr)))

    def _valid(self):
        if self.ptr == None:
            raise TractError("invalid value (maybe already consumed ?)")

    def from_numpy(array: numpy.ndarray) -> "Value":
        array = numpy.ascontiguousarray(array)

        data = array.__array_interface__['data'][0]
        data = c_void_p(data)
        ptr = c_void_p()
        shape_t = c_size_t * array.ndim
        shape = shape_t()
        for ix in range(0, array.ndim):
            shape[ix] = array.shape[ix]
        dt = dt_numpy_to_tract(array.dtype)
        check(lib.tract_value_from_bytes(dt, c_size_t(array.ndim), shape, data, byref(ptr)))
        return Value(ptr)

    def to_numpy(self) -> numpy.array:
        """Builds a numpy array equivalent to the data in this value."""
        self._valid()
        rank = c_size_t();
        shape = POINTER(c_size_t)()
        dt = c_float
        data = POINTER(dt)()
        check(lib.tract_value_as_bytes(self.ptr, None, byref(rank), byref(shape), byref(data)))
        rank = rank.value
        shape = [ int(shape[ix]) for ix in range(0, rank) ]
        array = numpy.ctypeslib.as_array(data, shape).copy()
        return array

    def into_numpy(self) -> numpy.array:
        """Same as to_numpy(), but drop the value content once the numpy array is built."""
        result = self.to_numpy()
        check(lib.tract_value_destroy(byref(self.ptr)))
        return result

