"""com.microsoft GroupQueryAttention, one generated token against a 5-long KV cache.

do_rotary=1 with no position_ids, so the op has to place the new token at seqlens_k
(total_sequence_length - 1) and rotate Q and K before the cache is appended to.
Expected outputs come from onnxruntime.
"""
import numpy as np, onnx, onnxruntime
from onnx import helper, TensorProto

B, NUM_HEADS, KV_HEADS, HEAD_SIZE = 1, 4, 2, 16
SEQ, PAST, MAX_SEQ = 1, 5, 32
HALF, TOTAL = HEAD_SIZE // 2, PAST + SEQ

node = helper.make_node(
    "GroupQueryAttention",
    ["query", "key", "value", "past_key", "past_value", "seqlens_k",
     "total_sequence_length", "cos_cache", "sin_cache"],
    ["output", "present_key", "present_value"],
    domain="com.microsoft",
    num_heads=NUM_HEADS, kv_num_heads=KV_HEADS, local_window_size=-1,
    do_rotary=1, rotary_interleaved=0,
)
vi = helper.make_tensor_value_info
graph = helper.make_graph(
    [node], "gqa_decode_rope",
    [vi("query", TensorProto.FLOAT, [B, SEQ, NUM_HEADS * HEAD_SIZE]),
     vi("key", TensorProto.FLOAT, [B, SEQ, KV_HEADS * HEAD_SIZE]),
     vi("value", TensorProto.FLOAT, [B, SEQ, KV_HEADS * HEAD_SIZE]),
     vi("past_key", TensorProto.FLOAT, [B, KV_HEADS, PAST, HEAD_SIZE]),
     vi("past_value", TensorProto.FLOAT, [B, KV_HEADS, PAST, HEAD_SIZE]),
     vi("seqlens_k", TensorProto.INT32, [B]),
     vi("total_sequence_length", TensorProto.INT32, []),
     vi("cos_cache", TensorProto.FLOAT, [MAX_SEQ, HALF]),
     vi("sin_cache", TensorProto.FLOAT, [MAX_SEQ, HALF])],
    [vi("output", TensorProto.FLOAT, [B, SEQ, NUM_HEADS * HEAD_SIZE]),
     vi("present_key", TensorProto.FLOAT, [B, KV_HEADS, TOTAL, HEAD_SIZE]),
     vi("present_value", TensorProto.FLOAT, [B, KV_HEADS, TOTAL, HEAD_SIZE])],
)
model = helper.make_model(
    graph, opset_imports=[helper.make_opsetid("", 17),
                          helper.make_opsetid("com.microsoft", 1)])
model.ir_version = 10
onnx.save(model, "model.onnx")

rng = np.random.RandomState(20260817)
f = lambda *s: (rng.randn(*s) * 0.5).astype(np.float32)
angle = np.arange(MAX_SEQ, dtype=np.float32)[:, None] * (
    1.0 / 10000 ** (np.arange(HALF, dtype=np.float32) / HALF))[None, :]
inputs = dict(
    query=f(B, SEQ, NUM_HEADS * HEAD_SIZE),
    key=f(B, SEQ, KV_HEADS * HEAD_SIZE),
    value=f(B, SEQ, KV_HEADS * HEAD_SIZE),
    past_key=f(B, KV_HEADS, PAST, HEAD_SIZE),
    past_value=f(B, KV_HEADS, PAST, HEAD_SIZE),
    seqlens_k=np.full((B,), TOTAL - 1, dtype=np.int32),
    total_sequence_length=np.array(TOTAL, dtype=np.int32),
    cos_cache=np.cos(angle), sin_cache=np.sin(angle),
)

sess = onnxruntime.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
output, present_key, present_value = sess.run(None, inputs)
np.savez_compressed("io.npz", output=output, present_key=present_key,
                    present_value=present_value, **inputs)
