"""com.microsoft GatherBlockQuantized with a constant table, block size 32.

Same operator as ../gather_block_quantized, but the table, scales and zero points are
initializers rather than graph inputs, and the block size is 32 — which is what the
ORT-GenAI exports emit, and what lets tract keep the table as a Q4_0 block-quant constant
read by a plain Gather.

Q4_0 stores the scale as an f16, so the scales here are drawn f16-representable. That keeps
the whole case exact: tract computes (q - 8) * s + (8 - z) * s where onnxruntime computes
(q - z) * s, and with s exact in f16 both products and their sum are exact in f32. An export
with arbitrary f32 scales is off by the f16 rounding of the scale, which is the same
departure the MatMulNBits Q4_0 path already takes.
"""
import numpy as np, onnx, onnxruntime
from onnx import helper, numpy_helper, TensorProto

ROWS, COLS, BLOCK = 40, 64, 32
BLOCKS = COLS // BLOCK
INDICES = (2, 3)

rng = np.random.RandomState(20260909)
data = rng.randint(0, 256, (ROWS, COLS // 2), dtype=np.uint8)
scales = (rng.rand(ROWS, BLOCKS) * 0.05 + 0.01).astype(np.float16).astype(np.float32)
zero_points = rng.randint(0, 256, (ROWS, BLOCKS // 2), dtype=np.uint8)
indices = rng.randint(0, ROWS, INDICES).astype(np.int64)

node = helper.make_node(
    "GatherBlockQuantized",
    ["data", "indices", "scales", "zero_points"], ["output"],
    domain="com.microsoft",
    bits=4, block_size=BLOCK, gather_axis=0, quantize_axis=1,
)
vi = helper.make_tensor_value_info
graph = helper.make_graph(
    [node], "gather_block_quantized_const",
    [vi("indices", TensorProto.INT64, list(INDICES))],
    [vi("output", TensorProto.FLOAT, list(INDICES) + [COLS])],
    initializer=[numpy_helper.from_array(data, "data"),
                 numpy_helper.from_array(scales, "scales"),
                 numpy_helper.from_array(zero_points, "zero_points")],
)
model = helper.make_model(
    graph, opset_imports=[helper.make_opsetid("", 21),
                          helper.make_opsetid("com.microsoft", 1)])
model.ir_version = 10
onnx.save(model, "model.onnx")

sess = onnxruntime.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
output = sess.run(None, dict(indices=indices))[0]
np.savez_compressed("io.npz", output=output, indices=indices)
