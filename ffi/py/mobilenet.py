import tract
import numpy

nnef = tract.nnef()

mobilenet = nnef.model_for_path("mobilenet_v2_1.0.onnx.nnef.tgz")
mobilenet.optimize()
runnable = mobilenet.into_runnable()

img = numpy.load("grace_hopper_1x3x224x244.npy")
img = tract.Value.from_numpy(img)

result = runnable.run([img])
confidences = result[0].to_numpy()
print(confidences.shape)
print(confidences[:,0:10])
print(numpy.argmax(confidences))
