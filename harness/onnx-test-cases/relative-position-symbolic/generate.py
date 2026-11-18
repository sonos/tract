#!/usr/bin/env python3
"""Relative-position attention shape idioms (VITS, Supertonic) over a symbolic length L.

y: relative to absolute positions. The scores [L, 2L-1] are padded to [L, 2L], flattened to
2*L^2 (computed with Pow), padded by L-1 and reshaped to [L+1, 2L-1], then cropped to [L, L].
z: the relative embedding window. A [5, 2] table is padded by max(L-3, 0) on both sides, then
sliced from max(3-L, 0) to 2L-1 rows.
"""
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

HERE = Path(__file__).resolve().parent
i64 = lambda name, values: helper.make_tensor(name, TensorProto.INT64, [len(values)], values)
scalar = lambda name, value: helper.make_tensor(name, TensorProto.INT64, [], [value])
emb = np.arange(10, dtype=np.float32).reshape(5, 2)

nodes = [
    helper.make_node("Shape", ["x"], ["shape"]),
    helper.make_node("Gather", ["shape", "zero"], ["L"], axis=0),
    helper.make_node("Mul", ["L", "two"], ["L2"]),
    helper.make_node("Sub", ["L2", "one"], ["W"]),
    helper.make_node("Range", ["zero", "W", "one"], ["cols_i"]),
    helper.make_node("Cast", ["cols_i"], ["cols"], to=TensorProto.FLOAT),
    helper.make_node("Unsqueeze", ["x", "axis1"], ["rows"]),
    helper.make_node("Unsqueeze", ["cols", "axis0"], ["cols2"]),
    helper.make_node("Mul", ["rows", "hundred"], ["rows100"]),
    helper.make_node("Add", ["rows100", "cols2"], ["scores"]),
    helper.make_node("Pad", ["scores", "pad_last1"], ["padded"]),
    helper.make_node("Pow", ["L", "two"], ["Lsq"]),
    helper.make_node("Mul", ["Lsq", "two"], ["flat_len"]),
    helper.make_node("Unsqueeze", ["flat_len", "axis0"], ["flat_shape"]),
    helper.make_node("Reshape", ["padded", "flat_shape"], ["flat"]),
    helper.make_node("Sub", ["L", "one"], ["Lm1"]),
    helper.make_node("Unsqueeze", ["Lm1", "axis0"], ["Lm1v"]),
    helper.make_node("Concat", ["zero1", "Lm1v"], ["pads2"], axis=0),
    helper.make_node("Pad", ["flat", "pads2"], ["flat_padded"]),
    helper.make_node("Add", ["L", "one"], ["Lp1"]),
    helper.make_node("Unsqueeze", ["Lp1", "axis0"], ["Lp1v"]),
    helper.make_node("Unsqueeze", ["W", "axis0"], ["Wv"]),
    helper.make_node("Concat", ["Lp1v", "Wv"], ["square_shape"], axis=0),
    helper.make_node("Reshape", ["flat_padded", "square_shape"], ["square"]),
    helper.make_node("Unsqueeze", ["L", "axis0"], ["Lv"]),
    helper.make_node("Concat", ["zero1", "Lm1v"], ["starts"], axis=0),
    helper.make_node("Concat", ["Lv", "big1"], ["ends"], axis=0),
    helper.make_node("Slice", ["square", "starts", "ends"], ["y"]),
    helper.make_node("Sub", ["L", "three"], ["Lm3"]),
    helper.make_node("Max", ["Lm3", "zero"], ["pad_len"]),
    helper.make_node("Unsqueeze", ["pad_len", "axis0"], ["pad_lenv"]),
    helper.make_node("Concat", ["pad_lenv", "zero1", "pad_lenv", "zero1"], ["emb_pads"], axis=0),
    helper.make_node("Pad", ["emb", "emb_pads"], ["emb_padded"]),
    helper.make_node("Sub", ["three", "L"], ["three_m_L"]),
    helper.make_node("Max", ["three_m_L", "zero"], ["start"]),
    helper.make_node("Add", ["start", "W"], ["end"]),
    helper.make_node("Unsqueeze", ["start", "axis0"], ["startv"]),
    helper.make_node("Unsqueeze", ["end", "axis0"], ["endv"]),
    helper.make_node("Slice", ["emb_padded", "startv", "endv", "zero1"], ["z"]),
]
constants = [
    scalar("zero", 0), scalar("one", 1), scalar("two", 2), scalar("three", 3),
    i64("zero1", [0]), i64("big1", [(1 << 63) - 1]), i64("axis0", [0]), i64("axis1", [1]),
    i64("pad_last1", [0, 0, 0, 1]),
    helper.make_tensor("hundred", TensorProto.FLOAT, [], [100.0]),
    helper.make_tensor("emb", TensorProto.FLOAT, emb.shape, emb.ravel()),
]
model = helper.make_model(
    helper.make_graph(
        nodes, "relative_position_symbolic",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["L"])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["L", "L"]),
         helper.make_tensor_value_info("z", TensorProto.FLOAT, ["W", 2])],
        constants),
    opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
onnx.checker.check_model(model, full_check=True)
onnx.save(model, HERE / "model.onnx")


def reference(x):
    L = len(x)
    scores = x[:, None] * 100 + np.arange(2 * L - 1, dtype=np.float32)[None, :]
    flat = np.pad(scores, ((0, 0), (0, 1))).reshape(-1)
    square = np.pad(flat, (0, L - 1)).reshape(L + 1, 2 * L - 1)
    pad_len = max(L - 3, 0)
    start = max(3 - L, 0)
    return square[:L, L - 1:], np.pad(emb, ((pad_len, pad_len), (0, 0)))[start:start + 2 * L - 1]


for L in (2, 6):
    x = np.arange(1, L + 1, dtype=np.float32)
    y, z = reference(x)
    np.savez(HERE / f"io-{L}.npz", x=x, y=y, z=z)
