import numpy
from ctypes import *
from pathlib import Path
from typing import Dict, List, Union

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

def version() -> str:
    return str(lib.tract_version(), "utf-8")

def nnef() -> "Nnef":
    return Nnef()

def onnx() -> "Onnx":
    return Onnx()

def check(err):
    if err != 0:
        raise TractError(str(lib.tract_get_last_error(), "utf-8"))

class Nnef:
    def __init__(self):
        ptr = c_void_p()
        check(lib.tract_nnef_create(byref(ptr)))
        self.ptr = ptr

    def __del__(self):
        check(lib.tract_nnef_destroy(byref(self.ptr)))

    def _valid(self):
        if self.ptr == None:
            raise TractError("invalid inference model (maybe already consumed ?)")

    def valid(self):
        if self.ptr == None:
            raise TractError("invalid inference model (maybe already consumed ?)")

    def model_for_path(self, path: Union[str, Path]) -> "Model":
        self._valid()
        model = c_void_p()
        path = str(path).encode("utf-8")
        check(lib.tract_nnef_model_for_path(self.ptr, path, byref(model)))
        return Model(model)

    def with_tract_core(self) -> "Nnef":
        self._valid()
        check(lib.tract_nnef_enable_tract_core(self.ptr))
        return self

    def with_onnx(self) -> "Nnef":
        self._valid()
        check(lib.tract_nnef_enable_onnx(self.ptr))
        return self

    def with_pulse(self) -> "Nnef":
        self._valid()
        check(lib.tract_nnef_enable_pulse(self.ptr))
        return self

    def write_model_to_dir(self, model: "Model", path: Union[str, Path]) -> None:
        self._valid()
        model._valid()
        if not isinstance(model, Model):
            raise TractError("Expected a Model, called with " + model);
        path = str(path).encode("utf-8")
        print(path);
        check(lib.tract_nnef_write_model_to_dir(self.ptr, path, model.ptr))

    def write_model_to_tar(self, model: "Model", path: Union[str, Path]) -> None:
        self._valid()
        model._valid()
        if not isinstance(model, Model):
            raise TractError("Expected a Model, called with " + model);
        print(path)
        path = str(path).encode("utf-8")
        print(path);
        check(lib.tract_nnef_write_model_to_tar(self.ptr, path, model.ptr))

    def write_model_to_tar_gz(self, model: "Model", path: Union[str, Path]) -> None:
        self._valid()
        model._valid()
        if not isinstance(model, Model):
            raise TractError("Expected a Model, called with " + model);
        path = str(path).encode("utf-8")
        check(lib.tract_nnef_write_model_to_tar_gz(self.ptr, path, model.ptr))

class Onnx:
    def __init__(self):
        ptr = c_void_p()
        check(lib.tract_onnx_create(byref(ptr)))
        self.ptr = ptr

    def __del__(self):
        check(lib.tract_onnx_destroy(byref(self.ptr)))

    def model_for_path(self, path: Union[str, Path]) -> "InferenceModel":
        model = c_void_p()
        path = str(path).encode("utf-8")
        check(lib.tract_onnx_model_for_path(self.ptr, path, byref(model)))
        return InferenceModel(model)

class InferenceModel:
    def __init__(self, ptr):
        self.ptr = ptr

    def __del__(self):
        if self.ptr:
            check(lib.tract_inference_model_destroy(byref(self.ptr)))

    def _valid(self):
        if self.ptr == None:
            raise TractError("invalid inference model (maybe already consumed ?)")

    def into_optimized(self) -> "Model":
        self._valid()
        model = c_void_p()
        check(lib.tract_inference_model_into_optimized(byref(self.ptr), byref(model)))
        return Model(model)

    def into_typed(self) -> "Model":
        self._valid()
        model = c_void_p()
        check(lib.tract_inference_model_into_typed(byref(self.ptr), byref(model)))
        return Model(model)

    def input_count(self) -> int:
        self._valid()
        i = c_size_t()
        check(lib.tract_inference_model_nbio(self.ptr, byref(i), None))
        return i.value

    def output_count(self) -> int:
        self._valid()
        i = c_size_t()
        check(lib.tract_inference_model_nbio(self.ptr, None, byref(i)))
        return i.value

    def input_name(self, input_id: int) -> str:
        self._valid()
        cstring = c_char_p()
        check(lib.tract_inference_model_input_name(self.ptr, input_id, byref(cstring)))
        result = str(cstring.value, "utf-8")
        lib.tract_free_cstring(cstring)
        return result

    def input_fact(self, input_id: int) -> "InferenceFact":
        self._valid()
        fact = c_void_p()
        check(lib.tract_inference_model_input_fact(self.ptr, input_id, byref(fact)))
        return InferenceFact(fact)

    def set_input_fact(self, input_id: int, fact: Union["InferenceFact", str, None]) -> None:
        self._valid()
        if isinstance(fact, str):
            fact = self.fact(fact)
        if fact == None:
            check(lib.tract_inference_model_set_input_fact(self.ptr, input_id, None))
        else:
            check(lib.tract_inference_model_set_input_fact(self.ptr, input_id, fact.ptr))

    def output_name(self, output_id: int) -> str:
        self._valid()
        cstring = c_char_p()
        check(lib.tract_inference_model_output_name(self.ptr, output_id, byref(cstring)))
        result = str(cstring.value, "utf-8")
        lib.tract_free_cstring(cstring)
        return result

    def output_fact(self, output_id: int) -> "InferenceFact":
        self._valid()
        fact = c_void_p()
        check(lib.tract_inference_model_output_fact(self.ptr, output_id, byref(fact)))
        return InferenceFact(fact)

    def set_output_fact(self, output_id: int, fact: Union["InferenceFact", str, None]) -> None:
        self._valid()
        if isinstance(fact, str):
            fact = self.fact(fact)
        if fact == None:
            check(lib.tract_inference_model_set_output_fact(self.ptr, output_id, None))
        else:
            check(lib.tract_inference_model_set_output_fact(self.ptr, output_id, fact.ptr))

    def fact(self, spec:str) -> "InferenceFact":
        self._valid()
        spec = str(spec).encode("utf-8")
        fact = c_void_p();
        check(lib.tract_inference_fact_parse(self.ptr, spec, byref(fact)))
        return InferenceFact(fact)

    def analyse(self) -> None:
        self._valid()
        check(lib.tract_inference_model_analyse(self.ptr, False))

    def into_analysed(self) -> "InferenceModel":
        self.analyse()
        return self

class Model:
    def __init__(self, ptr):
        self.ptr = ptr

    def __del__(self):
        if self.ptr:
            check(lib.tract_model_destroy(byref(self.ptr)))

    def _valid(self):
        if self.ptr == None:
            raise TractError("invalid model (maybe already consumed ?)")

    def input_count(self) -> int:
        self._valid()
        i = c_size_t()
        check(lib.tract_model_nbio(self.ptr, byref(i), None))
        return i.value

    def output_count(self) -> int:
        self._valid()
        i = c_size_t()
        check(lib.tract_model_nbio(self.ptr, None, byref(i)))
        return i.value

    def input_name(self, input_id: int) -> str:
        self._valid()
        cstring = c_char_p()
        check(lib.tract_model_input_name(self.ptr, input_id, byref(cstring)))
        result = str(cstring.value, "utf-8")
        lib.tract_free_cstring(cstring)
        return result

    def input_fact(self, input_id: int) -> "Fact":
        self._valid()
        fact = c_void_p()
        check(lib.tract_model_input_fact(self.ptr, input_id, byref(fact)))
        return Fact(fact)

    def output_name(self, output_id: int) -> str:
        self._valid()
        cstring = c_char_p()
        check(lib.tract_model_output_name(self.ptr, output_id, byref(cstring)))
        result = str(cstring.value, "utf-8")
        lib.tract_free_cstring(cstring)
        return result

    def output_fact(self, input_id: int) -> "Fact":
        self._valid()
        fact = c_void_p()
        check(lib.tract_model_output_fact(self.ptr, input_id, byref(fact)))
        return Fact(fact)

    def concretize_symbols(self, values: Dict[str, int]) -> None:
        self._valid()
        nb = len(values)
        names_str = []
        names = (c_char_p * nb)()
        values_list = (c_int64 * nb)()
        for ix, (k, v) in enumerate(values.items()):
            names_str.append(str(k).encode("utf-8"))
            names[ix] = names_str[ix]
            values_list[ix] = v
        check(lib.tract_model_concretize_symbols(self.ptr, c_size_t(nb), names, values_list))

    def pulse(self, symbol: str, pulse: Union[str, int]) -> None:
        self._valid()
        check(lib.tract_model_pulse_simple(byref(self.ptr), symbol.encode("utf-8"), str(pulse).encode("utf-8")))

    def declutter(self) -> None:
        self._valid()
        check(lib.tract_model_declutter(self.ptr))

    def into_decluttered(self) -> "Model":
        self.declutter();
        return self

    def optimize(self) -> None:
        self._valid()
        check(lib.tract_model_optimize(self.ptr))

    def into_optimized(self) -> "Model":
        self.optimize()
        return self

    def into_runnable(self) -> "Runnable":
        self._valid()
        runnable = c_void_p()
        check(lib.tract_model_into_runnable(byref(self.ptr), byref(runnable)))
        return Runnable(runnable)

    def property_keys(self) -> List[str]:
        self._valid()
        count = c_size_t()
        check(lib.tract_model_property_count(self.ptr, byref(count)))
        count = count.value
        cstrings = (POINTER(c_char) * count)()
        check(lib.tract_model_property_names(self.ptr, cstrings))
        names = []
        for i in range(0, count):
            names.append(str(cast(cstrings[i], c_char_p).value, "utf-8"))
            lib.tract_free_cstring(cstrings[i])
        return names

    def property(self, name: str) -> "Value":
        self._valid()
        value = c_void_p()
        check(lib.tract_model_property(self.ptr, str(name).encode("utf-8"), byref(value)))
        return Value(value)

    def profile_json(self, inputs: Union[None, List[Union["Value", numpy.ndarray]]]) -> str:
        self._valid()
        cstring = c_char_p()
        input_values = []
        input_ptrs = None
        if inputs != None:
            for v in inputs:
                if isinstance(v, Value):
                    input_values.append(v)
                elif isinstance(v, numpy.ndarray):
                    input_values.append(Value.from_numpy(v))
                else:
                    raise TractError(f"Inputs must be of type tract.Value or numpy.Array, got {v}")
            input_ptrs = (c_void_p * len(inputs))()
            for ix, v in enumerate(input_values):
                input_ptrs[ix] = v.ptr
        check(lib.tract_model_profile_json(self.ptr, input_ptrs, byref(cstring)))
        result = str(cstring.value, "utf-8")
        lib.tract_free_cstring(cstring)
        return result


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

    def _valid(self):
        if self.ptr == None:
            raise TractError("invalid runnable (maybe already consumed ?)")

    def run(self, inputs: list[Union["Value", numpy.ndarray]]) -> list["Value"]:
        self._valid()
        input_values = []
        for v in inputs:
            if isinstance(v, Value):
                input_values.append(v)
            elif isinstance(v, numpy.ndarray):
                input_values.append(Value.from_numpy(v))
            else:
                raise TractError(f"Inputs must be of type tract.Value or numpy.Array, got {v}")
        input_ptrs = (c_void_p * self.inputs)()
        output_ptrs = (c_void_p * self.outputs)()
        for ix, v in enumerate(input_values):
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
        check(lib.tract_value_create(dt, c_size_t(array.ndim), shape, data, byref(ptr)))
        return Value(ptr)

    def to_numpy(self) -> numpy.array:
        self._valid()
        rank = c_size_t();
        shape = POINTER(c_size_t)()
        dt = c_float
        data = POINTER(dt)()
        check(lib.tract_value_inspect(self.ptr, None, byref(rank), byref(shape), byref(data)))
        rank = rank.value
        shape = [ int(shape[ix]) for ix in range(0, rank) ]
        array = numpy.ctypeslib.as_array(data, shape).copy()
        return array

    def into_numpy(self) -> numpy.array:
        result = self.to_numpy()
        check(lib.tract_value_destroy(byref(self.ptr)))
        return result

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

