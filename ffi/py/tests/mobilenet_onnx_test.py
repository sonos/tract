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
