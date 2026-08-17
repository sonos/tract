"""com.microsoft MatMulNBits at bits=2, the width the ternary ORT-GenAI exports use.

Block size 128 spans several of the 32-wide blocks tract's block-quant formats use, and the
zero points are packed four to a byte, so this exercises both the wider block and the 2-bit
zero point unpacking. Expected outputs come from onnxruntime.
"""
import numpy as np, onnx, onnxruntime
from onnx import helper, TensorProto, numpy_helper

M, K, N, BLOCK, BITS = 3, 256, 64, 128, 2
BLOCKS, PER_BYTE = K // BLOCK, 8 // BITS

rng = np.random.RandomState(20260817)
B = rng.randint(0, 256, (N, BLOCKS, BLOCK // PER_BYTE), dtype=np.uint8)
# Powers of two, so the packer's f16 scale is exact and every product below is too.
scales = np.float32(2.0) ** rng.randint(-6, -3, N * BLOCKS).astype(np.float32)
zero_points = rng.randint(0, 256, (N * ((BLOCKS + PER_BYTE - 1) // PER_BYTE),), dtype=np.uint8)

node = helper.make_node("MatMulNBits", ["A", "B", "scales", "zero_points"], ["Y"],
                        domain="com.microsoft", K=K, N=N, bits=BITS, block_size=BLOCK)
vi = helper.make_tensor_value_info
graph = helper.make_graph(
    [node], "matmulnbits_2bit", [vi("A", TensorProto.FLOAT, [1, M, K])],
    [vi("Y", TensorProto.FLOAT, [1, M, N])],
    initializer=[numpy_helper.from_array(B, "B"),
                 numpy_helper.from_array(scales, "scales"),
                 numpy_helper.from_array(zero_points, "zero_points")])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21),
                                                helper.make_opsetid("com.microsoft", 1)])
model.ir_version = 10
onnx.save(model, "model.onnx")

# Small integers keep every product and partial sum exactly representable, so the
# comparison tests the unpacking rather than float accumulation order.
inputs = dict(A=rng.randint(-4, 5, (1, M, K)).astype(np.float32))
sess = onnxruntime.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
np.savez_compressed("io.npz", Y=sess.run(None, inputs)[0], **inputs)
