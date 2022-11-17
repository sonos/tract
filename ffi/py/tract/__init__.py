import numpy
from tract.tract import lib, ffi

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
        i = ffi.new("uintptr_t *");
        o = ffi.new("uintptr_t *");
        check(lib.tract_runnable_nbio(self.ptr[0], i, o))
        self.inputs = int(i[0])
        self.outputs = int(o[0])

    def __del__(self):
        check(lib.tract_runnable_release(self.ptr))

    def run(self, inputs):
        input_ptrs = ffi.new("TractValue* []", self.inputs)
        output_ptrs = ffi.new("TractValue* []", self.outputs)
        for ix, v in enumerate(inputs):
            input_ptrs[ix] = v.ptr[0]
        check(lib.tract_runnable_run(self.ptr[0], input_ptrs, output_ptrs))
        result = []
        for ix in range(0, self.outputs):
            ptr = ffi.new("TractValue**", output_ptrs[ix])
            result.append(Value(ptr))
        return result

class Value:
    def __init__(self, ptr):
        self.ptr = ptr

    def __del__(self):
        check(lib.tract_value_destroy(self.ptr))

    def from_numpy(array):
        array = numpy.ascontiguousarray(array)
       
        data = array.__array_interface__['data'][0]
        data = ffi.cast("void*", data)
        ptr = ffi.new("TractValue **")
        check(lib.tract_value_create(lib.TRACT_DATUM_TYPE_F32, array.ndim, array.shape, data, ptr))
        return Value(ptr)

    def to_numpy(self):
        rank = ffi.new("uintptr_t*")
        shape = ffi.new("uintptr_t **")
        data = ffi.new("void **")
        check(lib.tract_value_inspect(self.ptr[0], ffi.NULL, rank, shape, data))
        rank = rank[0]
        shape = [ int(shape[0][ix]) for ix in range(0, rank) ]
        length = numpy.prod(shape)
        data = ffi.cast("float *", data[0])
        buffer = ffi.buffer(data, length * 4)
        array = numpy.frombuffer(buffer[:], dtype=numpy.float32) # force copy
        array = array.reshape(shape)
        return array

