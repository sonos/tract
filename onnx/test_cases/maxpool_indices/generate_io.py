"""MaxPool's Indices output over several (batch, channel) planes, in both storage orders.

ONNX flattens the index over the whole input tensor, so it includes the offset of the
(batch, channel) plane, and storage_order=1 makes the spatial part column-major. The
official tests only cover a single plane in row-major order. Expected outputs come from
onnxruntime.
"""
import numpy as np, onnx, onnxruntime
from onnx import helper, TensorProto

vi = helper.make_tensor_value_info
nodes = [
    helper.make_node("MaxPool", ["x"], ["y_row", "i_row"], kernel_shape=[2, 2], strides=[2, 2]),
    helper.make_node("MaxPool", ["x"], ["y_col", "i_col"], kernel_shape=[2, 2], strides=[2, 2],
                     storage_order=1),
]
graph = helper.make_graph(nodes, "maxpool_indices",
                          [vi("x", TensorProto.FLOAT, [2, 3, 4, 6])],
                          [vi("y_row", TensorProto.FLOAT, [2, 3, 2, 3]),
                           vi("i_row", TensorProto.INT64, [2, 3, 2, 3]),
                           vi("y_col", TensorProto.FLOAT, [2, 3, 2, 3]),
                           vi("i_col", TensorProto.INT64, [2, 3, 2, 3])])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
model.ir_version = 8
onnx.save(model, "model.onnx")

rng = np.random.RandomState(20260914)
# A permutation keeps every value distinct, so each window has a single argmax.
inputs = dict(x=rng.permutation(2 * 3 * 4 * 6).astype(np.float32).reshape(2, 3, 4, 6))
sess = onnxruntime.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
outputs = dict(zip(["y_row", "i_row", "y_col", "i_col"], sess.run(None, inputs)))
np.savez_compressed("io.npz", **outputs, **inputs)
