"""com.microsoft GatherBlockQuantized: an int4 block-quantized embedding lookup.

Mirrors what ORT-GenAI exports emit for a tied embedding table: uint8 storage holding two
4-bit values per byte (low nibble first), per-block f32 scales and nibble-packed zero
points. Expected outputs come from onnxruntime (needs a build new enough to register the
op; 1.19 is too old).
"""
import numpy as np, onnx, onnxruntime
from onnx import helper, TensorProto

ROWS, COLS, BLOCK = 40, 64, 32
BLOCKS = COLS // BLOCK
INDICES = (2, 3)

node = helper.make_node(
    "GatherBlockQuantized",
    ["data", "indices", "scales", "zero_points"], ["output"],
    domain="com.microsoft",
    bits=4, block_size=BLOCK, gather_axis=0, quantize_axis=1,
)
vi = helper.make_tensor_value_info
graph = helper.make_graph(
    [node], "gather_block_quantized",
    [vi("data", TensorProto.UINT8, [ROWS, COLS // 2]),
     vi("indices", TensorProto.INT64, list(INDICES)),
     vi("scales", TensorProto.FLOAT, [ROWS, BLOCKS]),
     vi("zero_points", TensorProto.UINT8, [ROWS, BLOCKS // 2])],
    [vi("output", TensorProto.FLOAT, list(INDICES) + [COLS])],
)
model = helper.make_model(
    graph, opset_imports=[helper.make_opsetid("", 21),
                          helper.make_opsetid("com.microsoft", 1)])
model.ir_version = 10
onnx.save(model, "model.onnx")

rng = np.random.RandomState(20260817)
inputs = dict(
    data=rng.randint(0, 256, (ROWS, COLS // 2), dtype=np.uint8),
    indices=rng.randint(0, ROWS, INDICES).astype(np.int64),
    scales=(rng.rand(ROWS, BLOCKS).astype(np.float32) * 0.05 + 0.01),
    zero_points=rng.randint(0, 256, (ROWS, BLOCKS // 2), dtype=np.uint8),
)

sess = onnxruntime.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
output = sess.run(None, inputs)[0]
np.savez_compressed("io.npz", output=output, **inputs)
