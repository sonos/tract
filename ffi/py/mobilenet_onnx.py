import tract
import numpy
import urllib.request

urllib.request.urlretrieve(
        "https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx",
        "mobilenetv2-7.onnx")

onnx = tract.onnx()

mobilenet = onnx.model_for_path("./mobilenetv2-7.onnx").into_optimized()
runnable = mobilenet.into_runnable()

img = numpy.load("grace_hopper_1x3x224x244.npy")

result = runnable.run([img])
confidences = result[0].to_numpy()
assert numpy.argmax(confidences) == 652
