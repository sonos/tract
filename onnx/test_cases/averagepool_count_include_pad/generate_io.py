"""AveragePool with count_include_pad=1, which divides by the full kernel.

The padded entries count towards the divisor but contribute nothing to the sum, so this
separates count_include_pad=1 from the default: here the window holds 4 real values out of
9, making the two conventions differ by 9/4. Expected outputs come from onnxruntime.
"""
import numpy as np, onnx, onnxruntime
from onnx import helper, TensorProto

node = helper.make_node("AveragePool", ["x"], ["y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1],
                        strides=[3, 3], ceil_mode=1, count_include_pad=1)
vi = helper.make_tensor_value_info
graph = helper.make_graph([node], "averagepool_count_include_pad",
                          [vi("x", TensorProto.FLOAT, [1, 3, 2, 2])],
                          [vi("y", TensorProto.FLOAT, [1, 3, 1, 1])])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 22)])
model.ir_version = 10
onnx.save(model, "model.onnx")

rng = np.random.RandomState(20260817)
inputs = dict(x=rng.rand(1, 3, 2, 2).astype(np.float32))
sess = onnxruntime.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
np.savez_compressed("io.npz", y=sess.run(None, inputs)[0], **inputs)
