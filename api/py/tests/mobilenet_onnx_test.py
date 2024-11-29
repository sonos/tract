import tract
import numpy
import urllib.request
import tempfile
import json
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
        .model_for_path("./mobilenetv2-7.onnx")
        .into_optimized()
        .into_runnable()
    )
    result = model.run([grace_hopper_1x3x224x244()])
    confidences = result[0].to_numpy()
    assert numpy.argmax(confidences) == 652

def test_state():
    model = (
        tract.onnx()
        .model_for_path("./mobilenetv2-7.onnx")
        .into_optimized()
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
        .model_for_path("mobilenet_v2_1.0.onnx.nnef.tgz")
        .into_optimized()
        .into_runnable()
    )
    result = model.run([grace_hopper_1x3x224x244()])
    confidences = result[0].to_numpy()
    assert numpy.argmax(confidences) == 652

def test_inference_model():
    model = tract.onnx().model_for_path("./mobilenetv2-7.onnx")
    assert model.input_count() == 1
    assert model.output_count() == 1
    assert model.input_name(0) == "data"
    assert model.output_name(0) == "mobilenetv20_output_flatten0_reshape0"
    assert str(model.input_fact(0)) == "1,3,224,224,F32"
    model.set_input_fact(0, "B,3,224,224,f32")
    model.set_output_fact(0, None)
    model.analyse()
    assert str(model.output_fact(0)) == "B,1000,F32"
    typed = model.into_typed()

def test_set_output_names_on_inference_model():
    model = tract.onnx().model_for_path("./mobilenetv2-7.onnx")
    model.set_input_fact(0, "B,3,224,224,f32")
    model.analyse()
    model.set_output_names(["mobilenetv20_output_pred_fwd"])
    assert str(model.output_fact(0)) == "B,1000,1,1,F32"

def test_typed_model():
    model = tract.nnef().model_for_path("mobilenet_v2_1.0.onnx.nnef.tgz")
    assert model.input_count() == 1
    assert model.output_count() == 1
    assert model.input_name(0) == "data"
    assert model.output_name(0) == "mobilenetv20_output_flatten0_reshape0"
    assert str(model.input_fact(0)) == "1,3,224,224,F32"
    assert str(model.output_fact(0)) == "1,1000,F32"
    model.declutter()

def test_set_output_names():
    model = tract.nnef().model_for_path("mobilenet_v2_1.0.onnx.nnef.tgz")
    model.set_output_names(["conv_53"])
    assert str(model.output_fact(0)) == "1,1000,1,1,F32"

def test_concretize():
    model = tract.onnx().model_for_path("./mobilenetv2-7.onnx")
    model.set_input_fact(0, "B,3,224,224,f32")
    model.analyse()
    typed = model.into_typed().into_decluttered()
    assert str(typed.input_fact(0)) == "B,3,224,224,F32"
    assert str(typed.output_fact(0)) == "B,1000,F32"
    typed.concretize_symbols({ "B": 1 })
    assert str(typed.input_fact(0)) == "1,3,224,224,F32"
    assert str(typed.output_fact(0)) == "1,1000,F32"

def test_pulse():
    model = tract.onnx().model_for_path("./mobilenetv2-7.onnx")
    model.set_input_fact(0, "B,3,224,224,f32")
    model.analyse()
    typed = model.into_typed().into_decluttered()
    assert str(typed.input_fact(0)) == "B,3,224,224,F32"
    assert str(typed.output_fact(0)) == "B,1000,F32"
    typed.pulse("B",  5)
    assert str(typed.input_fact(0)) == "5,3,224,224,F32"
    assert str(typed.output_fact(0)) == "5,1000,F32"
    properties = typed.property_keys()
    properties.sort()
    assert properties == ["pulse.delay", "pulse.input_axes", "pulse.output_axes"]
    assert typed.property("pulse.delay").to_numpy() == [0]

def test_f32_to_f16():
    model = tract.onnx().model_for_path("./mobilenetv2-7.onnx")
    model.set_input_fact(0, "1,3,224,224,f32")
    model.analyse()
    typed = model.into_typed().into_decluttered()
    typed.transform("f32-to-f16")
    assert str(typed.input_fact(0)) == "1,3,224,224,F16"
    assert str(typed.output_fact(0)) == "1,1000,F16"

def test_f16_to_f32():
    model = tract.onnx().model_for_path("./mobilenetv2-7.onnx")
    model.set_input_fact(0, "1,3,224,224,f32")
    model.analyse()
    
    #Convert model to half
    typed = model.into_typed().into_decluttered()
    typed.transform("f32-to-f16")
    assert str(typed.input_fact(0)) == "1,3,224,224,F16"
    assert str(typed.output_fact(0)) == "1,1000,F16"
    
    # Convert back to f32
    typed.transform("f16-to-f32")
    assert str(typed.input_fact(0)) == "1,3,224,224,F32"
    assert str(typed.output_fact(0)) == "1,1000,F32"

def test_typed_model_to_nnef_and_back():
    model = tract.onnx().model_for_path("./mobilenetv2-7.onnx")
    model.set_input_fact(0, "B,3,224,224,f32")
    model.analyse()
    typed = model.into_typed()
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = Path(tmpdirname)
        nnef = tract.nnef().with_tract_core()

        path = tmpdirname / "nnef-dir"
        nnef.write_model_to_dir(typed, path)
        reloaded = nnef.model_for_path(path)
        assert str(reloaded.input_fact(0)) == "B,3,224,224,F32"
        assert str(reloaded.output_fact(0)) == "B,1000,F32"

        path = tmpdirname / "nnef.tar"
        nnef.write_model_to_tar(typed, path)
        reloaded = nnef.model_for_path(path)
        assert str(reloaded.input_fact(0)) == "B,3,224,224,F32"
        assert str(reloaded.output_fact(0)) == "B,1000,F32"

        path = tmpdirname / "nnef.tar.gz"
        nnef = nnef.with_extended_identifier_syntax()
        nnef.write_model_to_tar_gz(typed, path)
        reloaded = nnef.model_for_path(path)
        assert str(reloaded.input_fact(0)) == "B,3,224,224,F32"
        assert str(reloaded.output_fact(0)) == "B,1000,F32"

def test_cost():
    model = tract.nnef().model_for_path("mobilenet_v2_1.0.onnx.nnef.tgz")
    assert str(model.input_fact(0)) == "1,3,224,224,F32"
    model.declutter()
    model.optimize()
    profile = model.profile_json(None)
    profile = json.loads(profile)
    assert len(profile["nodes"]) > 10
    assert profile["nodes"][0]["node_name"] != ""
    assert profile["nodes"][0]["op_name"] != ""
    assert next(filter(lambda node: "cost" in node and "FMA(F32)" in node["cost"], profile["nodes"]), None) != None

def test_profile():
    model = tract.nnef().model_for_path("mobilenet_v2_1.0.onnx.nnef.tgz")
    assert str(model.input_fact(0)) == "1,3,224,224,F32"
    model.declutter()
    model.optimize()
    data = numpy.random.rand(1,3,224,224).astype(dtype="float32")
    profile = model.profile_json([data])
    profile = json.loads(profile)
    profiling_info = profile["profiling_info"]
    assert profiling_info["iterations"] >= 1
    assert len(profile["nodes"]) > 10
    assert profile["nodes"][0]["node_name"] != ""
    assert profile["nodes"][0]["op_name"] != ""
    if "secs_per_iter" in profile["nodes"][0]:
        assert profile["nodes"][0]["secs_per_iter"] >= 0
    assert next(filter(lambda node: "cost" in node and "FMA(F32)" in node["cost"], profile["nodes"]), None) != None
