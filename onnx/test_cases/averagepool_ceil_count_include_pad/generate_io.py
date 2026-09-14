"""AveragePool with ceil_mode=1 and count_include_pad=1, where the last window overhangs.

With a 6x6 input, 3x3 kernel, stride 2 and one pixel of padding, ceil mode adds a fourth
window per axis that starts on the last input row/column and runs one tap past the padding.
The divisor counts the padding but not that overhang, so it is 6 on the last row and column
and 4 in the corner instead of 9. Expected outputs come from onnxruntime.
"""
import numpy as np, onnx, onnxruntime
from onnx import helper, TensorProto

node = helper.make_node("AveragePool", ["x"], ["y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1],
                        strides=[2, 2], ceil_mode=1, count_include_pad=1)
vi = helper.make_tensor_value_info
graph = helper.make_graph([node], "averagepool_ceil_count_include_pad",
                          [vi("x", TensorProto.FLOAT, [1, 2, 6, 6])],
                          [vi("y", TensorProto.FLOAT, [1, 2, 4, 4])])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
model.ir_version = 9
onnx.save(model, "model.onnx")

rng = np.random.RandomState(20260914)
inputs = dict(x=rng.rand(1, 2, 6, 6).astype(np.float32))
sess = onnxruntime.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
np.savez_compressed("io.npz", y=sess.run(None, inputs)[0], **inputs)
