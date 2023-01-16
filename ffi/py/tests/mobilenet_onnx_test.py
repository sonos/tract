import tract
import numpy
import urllib.request
from os import path

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
