import tract
import numpy
import urllib.request
import tempfile
import json
import pytest
from pathlib import Path

def setup_module(module):
    if not Path("mobilenetv2-7.onnx").exists():
        urllib.request.urlretrieve(
            "https://s3.amazonaws.com/tract-ci-builds/tests/mobilenetv2-7.onnx",
            "mobilenetv2-7.onnx",
        )
    if not Path("mobilenet_v2_1.0.onnx.nnef.tgz").exists():
        urllib.request.urlretrieve(
            "https://s3.amazonaws.com/tract-ci-builds/tests/mobilenet_v2_1.0.onnx.nnef.tgz",
            "mobilenet_v2_1.0.onnx.nnef.tgz"
        )

def grace_hopper_1x3x224x244():
    return numpy.load(Path(__file__).parent.parent / "grace_hopper_1x3x224x244.npy")

def test_version():
    tract.version()

def test_onnx():
    model = (
        tract.onnx()
        .load("./mobilenetv2-7.onnx")
        .into_tract()
        .into_runnable()
    )
    result = model.run([grace_hopper_1x3x224x244()])
    confidences = result[0].to_numpy()
    assert numpy.argmax(confidences) == 652

def test_state():
    model = (
        tract.onnx()
        .load("./mobilenetv2-7.onnx")
        .into_tract()
        .into_runnable()
    )
    state = model.spawn_state()
    result = state.run([grace_hopper_1x3x224x244()])
    confidences = result[0].to_numpy()
    assert numpy.argmax(confidences) == 652

def test_nnef_register():
    tract.nnef().with_tract_core().with_onnx().with_pulse().with_tract_extra()

def test_nnef():
    model = (
        tract.nnef()
        .load("mobilenet_v2_1.0.onnx.nnef.tgz")
        .into_runnable()
    )
    result = model.run([grace_hopper_1x3x224x244()])
    confidences = result[0].to_numpy()
    assert numpy.argmax(confidences) == 652

def test_inference_model():
    model = tract.onnx().load("./mobilenetv2-7.onnx")
    assert model.input_count() == 1
    assert model.output_count() == 1
    assert model.input_name(0) == "data"
    assert model.output_name(0) == "mobilenetv20_output_flatten0_reshape0"
    assert str(model.input_fact(0)) == "1,3,224,224,F32"
    model.set_input_fact(0, "B,3,224,224,f32")
    model.set_output_fact(0, None)
    model.analyse()
    assert str(model.output_fact(0)) == "B,1000,F32"
    typed = model.into_tract()

def test_set_output_names_on_inference_model():
    model = tract.onnx().load("./mobilenetv2-7.onnx")
    model.set_input_fact(0, "B,3,224,224,f32")
    model.analyse()
    model.set_output_names(["mobilenetv20_output_pred_fwd"])
    assert str(model.output_fact(0)) == "B,1000,1,1,F32"

def test_typed_model():
    model = tract.nnef().load("mobilenet_v2_1.0.onnx.nnef.tgz")
    assert model.input_count() == 1
    assert model.output_count() == 1
    assert model.input_name(0) == "data"
    assert model.output_name(0) == "conv_53"
    assert str(model.input_fact(0)) == "1,3,224,224,F32"
    assert str(model.output_fact(0)) == "1,1000,F32"

def test_runtime():
    model = tract.nnef().load("mobilenet_v2_1.0.onnx.nnef.tgz")
    rt = tract.runtime_for_name("default")
    runnable = rt.prepare(model)
    result = runnable.run([grace_hopper_1x3x224x244()])
    confidences = result[0].to_numpy()
    assert numpy.argmax(confidences) == 652

def test_set_output_names():
    model = tract.nnef().load("mobilenet_v2_1.0.onnx.nnef.tgz")
    model.set_output_names(["conv_53"])
    assert str(model.output_fact(0)) == "1,1000,F32"

def test_concretize():
    model = tract.onnx().load("./mobilenetv2-7.onnx")
    model.set_input_fact(0, "B,3,224,224,f32")
    model.analyse()
    typed = model.into_tract()
    assert str(typed.input_fact(0)) == "B,3,224,224,F32"
    assert str(typed.output_fact(0)) == "B,1000,F32"
    typed.concretize_symbols({ "B": 1 })
    assert str(typed.input_fact(0)) == "1,3,224,224,F32"
    assert str(typed.output_fact(0)) == "1,1000,F32"

def test_pulse():
    model = tract.onnx().load("./mobilenetv2-7.onnx")
    model.set_input_fact(0, "B,3,224,224,f32")
    model.analyse()
    typed = model.into_tract()
    assert str(typed.input_fact(0)) == "B,3,224,224,F32"
    assert str(typed.output_fact(0)) == "B,1000,F32"
    typed.pulse("B",  5)
    assert str(typed.input_fact(0)) == "5,3,224,224,F32"
    assert str(typed.output_fact(0)) == "5,1000,F32"
    properties = typed.property_keys()
    properties.sort()
    assert properties == ["pulse.delay", "pulse.input_axes", "pulse.output_axes"]
    assert typed.property("pulse.delay").to_numpy() == [0]

def test_runtime_fact():
    runnable = tract.nnef().load("mobilenet_v2_1.0.onnx.nnef.tgz").into_runnable()
    assert str(runnable.input_fact(0)) ==  "1,3,224,224,F32"
    assert str(runnable.output_fact(0)) == "1,1000,F32"

def test_runtime_properties():
    model = tract.onnx().load("./mobilenetv2-7.onnx")
    model.set_input_fact(0, "B,3,224,224,f32")
    model.analyse()
    typed = model.into_tract()
    typed.pulse("B", "5")
    runnable = typed.into_runnable()
    properties = runnable.property_keys()
    properties.sort()
    assert properties == ["pulse.delay", "pulse.input_axes", "pulse.output_axes"]
    assert runnable.property("pulse.delay").to_numpy() == [0]

def test_f32_to_f16():
    model = tract.onnx().load("./mobilenetv2-7.onnx")
    model.set_input_fact(0, "1,3,224,224,f32")
    model.analyse()
    typed = model.into_tract()
    typed.transform("f32-to-f16")
    assert str(typed.input_fact(0)) == "1,3,224,224,F16"
    assert str(typed.output_fact(0)) == "1,1000,F16"

def test_f16_to_f32():
    model = tract.onnx().load("./mobilenetv2-7.onnx")
    model.set_input_fact(0, "1,3,224,224,f32")
    model.analyse()
    
    #Convert model to half
    typed = model.into_tract()
    typed.transform("f32-to-f16")
    assert str(typed.input_fact(0)) == "1,3,224,224,F16"
    assert str(typed.output_fact(0)) == "1,1000,F16"
    
    # Convert back to f32
    typed.transform("f16-to-f32")
    assert str(typed.input_fact(0)) == "1,3,224,224,F32"
    assert str(typed.output_fact(0)) == "1,1000,F32"

def test_typed_model_to_nnef_and_back():
    model = tract.onnx().load("./mobilenetv2-7.onnx")
    model.set_input_fact(0, "B,3,224,224,f32")
    model.analyse()
    typed = model.into_tract()
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = Path(tmpdirname)
        nnef = tract.nnef().with_tract_core()

        path = tmpdirname / "nnef-dir"
        nnef.write_model_to_dir(typed, path)
        reloaded = nnef.load(path)
        assert str(reloaded.input_fact(0)) == "B,3,224,224,F32"
        assert str(reloaded.output_fact(0)) == "B,1000,F32"

        path = tmpdirname / "nnef.tar"
        nnef.write_model_to_tar(typed, path)
        reloaded = nnef.load(path)
        assert str(reloaded.input_fact(0)) == "B,3,224,224,F32"
        assert str(reloaded.output_fact(0)) == "B,1000,F32"

        path = tmpdirname / "nnef.tar.gz"
        nnef = nnef.with_extended_identifier_syntax()
        nnef.write_model_to_tar_gz(typed, path)
        reloaded = nnef.load(path)
        assert str(reloaded.input_fact(0)) == "B,3,224,224,F32"
        assert str(reloaded.output_fact(0)) == "B,1000,F32"

def test_cost():
    model = tract.nnef().load("mobilenet_v2_1.0.onnx.nnef.tgz")
    assert str(model.input_fact(0)) == "1,3,224,224,F32"
    runnable = model.into_runnable()
    profile = runnable.profile_json(None, None)
    profile = json.loads(profile)
    assert len(profile["nodes"]) > 10
    assert profile["nodes"][0]["node_name"] != ""
    assert profile["nodes"][0]["op_name"] != ""
    assert next(filter(lambda node: "cost" in node and "FMA(F32)" in node["cost"], profile["nodes"]), None) != None

def test_profile():
    model = tract.nnef().load("mobilenet_v2_1.0.onnx.nnef.tgz")
    assert str(model.input_fact(0)) == "1,3,224,224,F32"
    runnable = model.into_runnable()
    data = numpy.random.rand(1,3,224,224).astype(dtype="float32")
    profile = runnable.profile_json([data], None)
    profile = json.loads(profile)
    profiling_info = profile["profiling_info"]
    assert profiling_info["iterations"] >= 1
    assert len(profile["nodes"]) > 10
    assert profile["nodes"][0]["node_name"] != ""
    assert profile["nodes"][0]["op_name"] != ""
    if "secs_per_iter" in profile["nodes"][0]:
        assert profile["nodes"][0]["secs_per_iter"] >= 0
    assert next(filter(lambda node: "cost" in node and "FMA(F32)" in node["cost"], profile["nodes"]), None) != None

def test_transform_registry():
    nnef = tract.nnef().with_tract_core()
    model = nnef.load("mobilenet_v2_1.0.onnx.nnef.tgz")

    #Convert model to half
    model.transform("f32-to-f16")
    assert str(model.input_fact(0)) == "1,3,224,224,F16"
    assert str(model.output_fact(0)) == "1,1000,F16"
    
    # Convert back to f32 
    model.transform("f16-to-f32")
    assert str(model.input_fact(0)) == "1,3,224,224,F32"

def test_fact_and_dims():
    nnef = tract.nnef().with_tract_core()
    model = nnef.load("mobilenet_v2_1.0.onnx.nnef.tgz")
    fact = model.parse_fact("B,S+P,64,f32")
    assert fact.datum_type() == tract.DatumType.F32
    assert fact.rank() == 3
    assert str(fact.dim(1)) == "S+P"
    s_plus_p = fact.dim(1)
    s_plus_twelve = s_plus_p.eval({ "P": 12 })
    assert str(s_plus_twelve) == "S+12"
    fourteen = s_plus_twelve.eval({"S": 2})
    assert fourteen.to_int64() == 14
    assert int(fourteen) == 14

def test_fact_and_dims_iterators():
    nnef = tract.nnef().with_tract_core()
    model = nnef.load("mobilenet_v2_1.0.onnx.nnef.tgz")
    facts = model.input_facts()
    assert len(facts) == 1
    dims = facts[0].dims()
    assert len(dims) == 4
    assert int(dims[0]) == 1
    assert int(dims[1]) == 3
    assert int(dims[2]) == 224
    assert int(dims[3]) == 224

def test_runtime_fact_iterator():
    nnef = tract.nnef().with_tract_core()
    runnable = nnef.load("mobilenet_v2_1.0.onnx.nnef.tgz").into_runnable()
    inputs = runnable.input_facts();
    assert len(inputs) == 1
    assert str(inputs[0]) == "1,3,224,224,F32"
    outputs = runnable.output_facts();
    assert len(outputs) == 1
    assert str(outputs[0]) == "1,1000,F32"

def test_value_method():
    floats = tract.Value.from_numpy(numpy.array([-1, -0.3, 0., 0.25, 0.75, 1.2], dtype=numpy.float32))
    assert floats.datum_type().is_float()
    ints = floats.convert_to(tract.DatumType.I8)
    assert ints.datum_type().is_signed()
    assert numpy.array_equal(ints.to_numpy(), [-1, 0, 0, 0, 0, 1])
    same = tract.Value.from_numpy(numpy.array([-1, -0.3, 0., 0.25, 0.75, 1.2], dtype=numpy.float32))
    assert floats == same

# @pytest.mark.skip(reason="Model need to be downlaoded locally (use .travis/test-llm.sh)")
# def test_state_init():
#     nnef = tract.nnef().with_tract_core().with_tract_transformers()
#     model = nnef.load("TinyLlama--TinyLlama_v1.1-q40ef32.nnef.tgz")

#     # Do KV Cache optim
#     nnef.transform_model(model, "detect-kv-cache")
#     assert model.input_count() == 1

#     state = model.into_runnable().spawn_state()

#     state_facts = state.get_states_facts()
#     state_initializers = []
#     for fact in state_facts:
#         parts = str(fact).split(',')
#         dtype_str = parts.pop()
#         assert dtype_str == "F32"

#         dims = [int(p) if p.isdigit() else 4 for p in parts]

#         state_initializers.append(numpy.zeros(dims, dtype=numpy.float32))

#     state.set_states(state_initializers)
#     out_states = state.get_states()

#     for (ix, v) in enumerate(state_initializers):
#         assert numpy.all(out_states[ix].to_numpy() == v)

# @pytest.mark.skip(reason="Model need to be downlaoded locally (use .travis/test-llm.sh)")
# def test_profile_with_init_state():
#     nnef = tract.nnef().with_tract_core().with_tract_transformers()
#     model = nnef.load("TinyLlama--TinyLlama_v1.1-q40ef32.nnef.tgz")

#     input = numpy.random.rand(1,1).astype(dtype="int64")
#     state_initializers = [
#         numpy.random.rand(1, 4, 4, 64).astype("float32")
#         for _ in range(1, model.input_count())
#     ]

#     # Do KV Cache optim
#     nnef.transform_model(model, "detect-kv-cache")
#     assert model.input_count() == 1

#     model.profile_json([input], state_initializers)
