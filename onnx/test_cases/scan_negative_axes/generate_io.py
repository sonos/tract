"""Scan with negative scan_input_axes and scan_output_axes.

A negative axis counts from the back of the full scanned tensor, one rank above the body
slice. Here the input is scanned along its last axis and the output is stacked along its last
axis. Expected outputs come from onnxruntime.
"""
import numpy as np, onnx, onnxruntime
from onnx import helper, TensorProto

vi = helper.make_tensor_value_info
body = helper.make_graph(
    [
        helper.make_node("Add", ["s_in", "x_t"], ["s_out"]),
        helper.make_node("Identity", ["s_out"], ["y_t"]),
    ],
    "body",
    [vi("s_in", TensorProto.FLOAT, [2]), vi("x_t", TensorProto.FLOAT, [2])],
    [vi("s_out", TensorProto.FLOAT, [2]), vi("y_t", TensorProto.FLOAT, [2])],
)
node = helper.make_node("Scan", ["s0", "xs"], ["s", "ys"], body=body, num_scan_inputs=1,
                        scan_input_axes=[-1], scan_output_axes=[-1])
graph = helper.make_graph([node], "scan_negative_axes",
                          [vi("s0", TensorProto.FLOAT, [2]), vi("xs", TensorProto.FLOAT, [2, 3])],
                          [vi("s", TensorProto.FLOAT, [2]), vi("ys", TensorProto.FLOAT, [2, 3])])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
model.ir_version = 7
onnx.save(model, "model.onnx")

rng = np.random.RandomState(20260914)
inputs = dict(s0=rng.rand(2).astype(np.float32), xs=rng.rand(2, 3).astype(np.float32))
sess = onnxruntime.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
s, ys = sess.run(None, inputs)
np.savez_compressed("io.npz", s=s, ys=ys, **inputs)
