"""Standard-activation ONNX LSTMs with constant weights, the shape tract runs as LstmSeq.

Four independent LSTMs share the graph: batch 3, bidirectional, hidden 72 (above the size where
the input-side product moves into the graph), and a 3-step sequence, which stays a Scan. Each
emits Y, Y_h and Y_c. Expected outputs come from onnxruntime.
"""
import numpy as np, onnx, onnxruntime
from onnx import helper, numpy_helper, TensorProto

vi = helper.make_tensor_value_info
rng = np.random.default_rng(0)
cases = [  # name, seq, batch, input, hidden, direction
    ("batch", 12, 3, 8, 16, "forward"),
    ("bidi", 12, 1, 8, 16, "bidirectional"),
    ("wide", 6, 1, 8, 72, "forward"),
    ("short", 3, 1, 8, 16, "forward"),
]
nodes, inputs, outputs, initializers, feeds = [], [], [], [], {}
for name, t, b, i, h, direction in cases:
    d = 2 if direction == "bidirectional" else 1
    w = (rng.standard_normal((d, 4 * h, i)) / np.sqrt(i)).astype(np.float32)
    r = (rng.standard_normal((d, 4 * h, h)) / np.sqrt(h)).astype(np.float32)
    bias = (rng.standard_normal((d, 8 * h)) * 0.1).astype(np.float32)
    initializers += [
        numpy_helper.from_array(w, f"{name}_w"),
        numpy_helper.from_array(r, f"{name}_r"),
        numpy_helper.from_array(bias, f"{name}_b"),
    ]
    ys = [f"{name}_y", f"{name}_y_h", f"{name}_y_c"]
    nodes.append(
        helper.make_node(
            "LSTM",
            [f"{name}_x", f"{name}_w", f"{name}_r", f"{name}_b"],
            ys,
            hidden_size=h,
            direction=direction,
        )
    )
    inputs.append(vi(f"{name}_x", TensorProto.FLOAT, [t, b, i]))
    outputs += [
        vi(ys[0], TensorProto.FLOAT, [t, d, b, h]),
        vi(ys[1], TensorProto.FLOAT, [d, b, h]),
        vi(ys[2], TensorProto.FLOAT, [d, b, h]),
    ]
    feeds[f"{name}_x"] = rng.standard_normal((t, b, i)).astype(np.float32)

graph = helper.make_graph(nodes, "lstm_seq", inputs, outputs, initializer=initializers)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
model.ir_version = 8
onnx.save(model, "model.onnx")

sess = onnxruntime.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
names = [o.name for o in sess.get_outputs()]
np.savez_compressed("io.npz", **dict(zip(names, sess.run(None, feeds))), **feeds)
