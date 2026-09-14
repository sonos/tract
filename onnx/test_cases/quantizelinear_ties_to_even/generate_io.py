"""QuantizeLinear on values that land exactly halfway between two integers.

ONNX rounds x / y_scale to nearest, ties to even, so 0.5 and 2.5 go down while 1.5 and 3.5
go up, and the negative ties mirror that. Rounding ties away from zero gets half of them
wrong. Expected outputs come from onnxruntime.
"""
import numpy as np, onnx, onnxruntime
from onnx import helper, numpy_helper, TensorProto

vi = helper.make_tensor_value_info
nodes = [
    helper.make_node("QuantizeLinear", ["x", "scale", "zp_u8"], ["y_u8"]),
    helper.make_node("QuantizeLinear", ["x", "scale", "zp_i8"], ["y_i8"]),
]
graph = helper.make_graph(
    nodes,
    "quantizelinear_ties_to_even",
    [vi("x", TensorProto.FLOAT, [8])],
    [vi("y_u8", TensorProto.UINT8, [8]), vi("y_i8", TensorProto.INT8, [8])],
    initializer=[
        numpy_helper.from_array(np.array(0.5, np.float32), "scale"),
        numpy_helper.from_array(np.array(128, np.uint8), "zp_u8"),
        numpy_helper.from_array(np.array(0, np.int8), "zp_i8"),
    ],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 7
onnx.save(model, "model.onnx")

inputs = dict(x=np.array([0.25, 0.75, 1.25, 1.75, -0.25, -0.75, -1.25, -1.75], np.float32))
sess = onnxruntime.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
y_u8, y_i8 = sess.run(None, inputs)
np.savez_compressed("io.npz", y_u8=y_u8, y_i8=y_i8, **inputs)
