import tract
import numpy
import urllib.request
from os import path
import tempfile

def setup_module(module):
    if not path.exists("mobilenetv2-7.onnx"):
        urllib.request.urlretrieve(
            "https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx",
            "mobilenetv2-7.onnx",
        )
    if not path.exists(""):
        urllib.request.urlretrieve(
            "https://sfo2.digitaloceanspaces.com/nnef-public/mobilenet_v2_1.0.onnx.nnef.tgz",
            "mobilenet_v2_1.0.onnx.nnef.tgz"
        )

def test_version():
    tract.version()

def test_onnx():
    model = (
        tract.onnx()
        .model_for_path("./mobilenetv2-7.onnx")
        .into_optimized()
        .into_runnable()
    )
    img = numpy.load("grace_hopper_1x3x224x244.npy")

    result = model.run([img])
    confidences = result[0].to_numpy()
    assert numpy.argmax(confidences) == 652

def test_nnef_register():
    tract.nnef().with_tract_core().with_onnx().with_pulse()

def test_nnef():
    model = (
        tract.nnef()
        .model_for_path("mobilenet_v2_1.0.onnx.nnef.tgz")
        .into_optimized()
        .into_runnable()
    )
    img = numpy.load("grace_hopper_1x3x224x244.npy")

    result = model.run([img])
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

def test_typed_model():
    model = tract.nnef().model_for_path("mobilenet_v2_1.0.onnx.nnef.tgz")
    assert model.input_count() == 1
    assert model.output_count() == 1
    assert model.input_name(0) == "data"
    assert model.output_name(0) == "mobilenetv20_output_flatten0_reshape0"
    assert str(model.input_fact(0)) == "1,3,224,224,F32"
    assert str(model.output_fact(0)) == "1,1000,F32"
    model.declutter()

def test_concretize():
    model = tract.onnx().model_for_path("./mobilenetv2-7.onnx")
    model.set_input_fact(0, "B,3,224,224,f32")
    model.set_output_fact(0, None)
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
    model.set_output_fact(0, None)
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

def test_typed_model_to_nnef_and_back():
    model = tract.onnx().model_for_path("./mobilenetv2-7.onnx")
    model.set_input_fact(0, "B,3,224,224,f32")
    model.set_output_fact(0, None)
    model.analyse()
    typed = model.into_typed()
    with tempfile.TemporaryDirectory() as tmpdirname:
        nnef = tract.nnef().with_tract_core()
        path = tmpdirname.join("model")
        nnef.write_model_to_dir(typed, path)
        reloaded = nnef.model_for_path(path)
        assert str(reloaded.input_fact(0)) == "B,3,224,224,F32"
        assert str(reloaded.output_fact(0)) == "B,1000,F32"
