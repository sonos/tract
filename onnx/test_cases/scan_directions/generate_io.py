"""Scan with scan_input_directions and scan_output_directions set to reverse.

The body accumulates the scanned slice into the state and emits two scan outputs, so there are
more scan outputs than scan inputs. The input is consumed back to front, the first output is
written front to back and the second one back to front. Expected outputs come from onnxruntime.
"""
import numpy as np, onnx, onnxruntime
from onnx import helper, TensorProto

vi = helper.make_tensor_value_info
body = helper.make_graph(
    [
        helper.make_node("Add", ["s_in", "x_t"], ["s_out"]),
        helper.make_node("Identity", ["s_out"], ["y_t"]),
        helper.make_node("Neg", ["s_out"], ["z_t"]),
    ],
    "body",
    [vi("s_in", TensorProto.FLOAT, [2]), vi("x_t", TensorProto.FLOAT, [2])],
    [vi("s_out", TensorProto.FLOAT, [2]), vi("y_t", TensorProto.FLOAT, [2]),
     vi("z_t", TensorProto.FLOAT, [2])],
)
node = helper.make_node("Scan", ["s0", "xs"], ["s", "ys", "zs"], body=body, num_scan_inputs=1,
                        scan_input_directions=[1], scan_output_directions=[0, 1])
graph = helper.make_graph([node], "scan_directions",
                          [vi("s0", TensorProto.FLOAT, [2]), vi("xs", TensorProto.FLOAT, [4, 2])],
                          [vi("s", TensorProto.FLOAT, [2]), vi("ys", TensorProto.FLOAT, [4, 2]),
                           vi("zs", TensorProto.FLOAT, [4, 2])])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
model.ir_version = 7
onnx.save(model, "model.onnx")

rng = np.random.RandomState(20260914)
inputs = dict(s0=rng.rand(2).astype(np.float32), xs=rng.rand(4, 2).astype(np.float32))
sess = onnxruntime.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
s, ys, zs = sess.run(None, inputs)
np.savez_compressed("io.npz", s=s, ys=ys, zs=zs, **inputs)
