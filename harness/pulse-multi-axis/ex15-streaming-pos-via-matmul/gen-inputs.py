#!/usr/bin/env python3
"""ex15-streaming-pos-via-matmul — io.npz generator.

Same skeleton as ex12 but with `pos` reached through a multi-input
matmul (`pos = matmul(pos_raw, pos_w)`) so blockify's
`trace_to_source` can't find a `Source` to mark session-buffered.
"""
import numpy as np

T, D, P = 4, 4, 2
rng = np.random.default_rng(15)
a       = rng.standard_normal((T, D)).astype(np.float32)
b       = rng.standard_normal((T, D)).astype(np.float32)
c_in    = rng.standard_normal((T, D)).astype(np.float32)
pos_raw = rng.standard_normal((2 * T - 1, D)).astype(np.float32)
pos_w   = rng.standard_normal((D, D)).astype(np.float32)

pos = pos_raw @ pos_w                              # [2T-1, D]


def skew(pos_raw_mat: np.ndarray) -> np.ndarray:
    T_in = pos_raw_mat.shape[0]
    padded = np.concatenate([np.zeros((T_in, 1), dtype=pos_raw_mat.dtype), pos_raw_mat], axis=1)
    view = padded.reshape(-1, T_in)
    sliced = view[1 : 2 * T_in, :]
    bd = sliced.reshape(T_in, -1)
    return bd[:, :T_in]


content    = a @ b.T
pos_einsum = a @ pos.T
pos_scores = skew(pos_einsum)
scores     = (content + pos_scores) * 0.5

idx       = np.arange(T)
chunk     = idx // P
diff      = chunk[:, None] - chunk[None, :]
in_block  = (diff >= 0) & (diff <= 1)             # banded-causal
masked    = np.where(in_block, scores, -np.inf)

m         = masked.max(axis=-1, keepdims=True)
exp_s     = np.exp(masked - m)
attn      = exp_s / exp_s.sum(axis=-1, keepdims=True)

output    = (attn @ c_in).astype(np.float32)

np.savez("io.npz", a=a, b=b, c=c_in, pos_raw=pos_raw, pos_w=pos_w, output=output)
print(f"Saved io.npz  a={a.shape} pos_raw={pos_raw.shape} pos_w={pos_w.shape} output={output.shape}")
