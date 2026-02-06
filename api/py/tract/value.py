import numpy
import math
from ctypes import *
from typing import Dict, List, Union
from tract.bindings import TractError, check, lib

from enum import IntEnum

class DatumType(IntEnum):
    BOOL = 0x01
    U8 = 0x11
    U16 = 0x12
    U32 = 0x14
    U64 = 0x18
    I8 = 0x21
    I16 = 0x22
    I32 = 0x24
    I64 = 0x28
    F16 = 0x32
    F32 = 0x34
    F64 = 0x38
    COMPLEX_I16 = 0x42
    COMPLEX_I32 = 0x44
    COMPLEX_I64 = 0x48
    COMPLEX_F16 = 0x52
    COMPLEX_F32 = 0x54
    COMPLEX_F64 = 0x58

    def __str__(self) -> str:
        return self.name

    def is_bool(self) -> bool:
        return self == self.BOOL

    def is_number(self) -> bool:
        return self != self.BOOL
        
    def is_float(self) -> bool:
        return self == self.F16 or self == self.F32 or self == self.F64

    def is_signed(self) -> bool:
        return self == self.I8 or self == self.I16 or self == self.I32 or self == self.I64

    def is_unsigned(self) -> bool:
        return self == self.U8 or self == self.U16 or self == self.U32 or self == self.U64

    def ctype(self):
        if self == self.BOOL:
            return c_bool
        if self == self.U8:
            return c_uint8
        if self == self.U16:
            return c_uint16
        if self == self.U32:
            return c_uint32
        if self == self.U64:
            return c_uint64
        if self == self.I8:
            return c_int8
        if self == self.I16:
            return c_int16
        if self == self.I32:
            return c_int32
        if self == self.I64:
            return c_int64        
        if self == self.F32:
            return c_float
        if self == self.F64:
            return c_double
        raise "invalid datum type"

def dt_numpy_to_tract(dt):
    if dt.kind == 'b':
        return DatumType.BOOL
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

    def __str__(self):
        return self.dump()
 
    def _valid(self):
        if self.ptr == None:
            raise TractError("invalid value (maybe already consumed ?)")

    def __eq__(self, other):
        (self_dt, self_shape, self_ptr) = self._parts()
        (other_dt, other_shape, other_ptr) = other._parts()
        self_len = math.prod(self_shape) * sizeof(self_dt.ctype())
        other_len = math.prod(self_shape) * sizeof(self_dt.ctype())
        self_buf = string_at(self_ptr, self_len) 
        other_buf = string_at(other_ptr, other_len)
        return self_dt == other_dt and self_shape == other_shape and self_buf == other_buf
        
        
    def _parts(self) -> (DatumType, [int], c_void_p):
        self._valid()
        rank = c_size_t();
        shape = POINTER(c_size_t)()
        data = c_void_p();
        dt = c_uint32(0)
        check(lib.tract_value_as_bytes(self.ptr, byref(dt), byref(rank), byref(shape), byref(data)))
        rank = rank.value
        shape = [ int(shape[ix]) for ix in range(0, rank) ]
        return (DatumType(dt.value), shape, data)

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
        (dt, shape, data) = self._parts()
        data = cast(data, POINTER(dt.ctype()))
        array = numpy.ctypeslib.as_array(data, shape).copy()
        return array

    def into_numpy(self) -> numpy.array:
        """Same as to_numpy(), but drop the value content once the numpy array is built."""
        result = self.to_numpy()
        check(lib.tract_value_destroy(byref(self.ptr)))
        return result

    def datum_type(self) -> DatumType:
        self._valid()
        dt = c_uint32(0)
        check(lib.tract_value_as_bytes(self.ptr, byref(dt), None, None, None))
        return DatumType(dt.value)

    def convert_to(self, to: DatumType) -> "Value":
        self._valid()
        ptr = c_void_p()
        check(lib.tract_value_convert_to(self.ptr, to, byref(ptr)))
        return Value(ptr)
        
    def dump(self):
        self._valid()
        cstring = c_char_p();
        check(lib.tract_value_dump(self.ptr, byref(cstring)))
        result = str(cstring.value, "utf-8")
        lib.tract_free_cstring(cstring)
        return result
