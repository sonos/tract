#!/usr/bin/env python3
"""Generate io.npz for ex08-batch-mask.

B=1, T=8, P=2, left_chunks=1, Dh=4
Input:  qkv [1, T, 12]
Output: [1, T, 4]
"""
import numpy as np

T, Dh, P, left_chunks, B = 8, 4, 2, 1, 1
rng = np.random.default_rng(42)

q = rng.standard_normal((B, T, Dh)).astype(np.float32)
k = rng.standard_normal((B, T, Dh)).astype(np.float32)
v = rng.standard_normal((B, T, Dh)).astype(np.float32)

chunk_idx = np.arange(T) // P
diff = chunk_idx[:, None] - chunk_idx[None, :]   # [T, T]
mask = (diff >= 0) & (diff <= left_chunks)        # [T, T]

scores = np.einsum("bid,bjd->bij", q, k)          # [B, T, T]
masked = np.where(mask[None], scores, -np.inf)

mx    = masked.max(axis=2, keepdims=True)
exp_s = np.exp(masked - mx)
attn  = exp_s / exp_s.sum(axis=2, keepdims=True)

output = np.einsum("bij,bjd->bid", attn, v).astype(np.float32)
qkv    = np.concatenate([q, k, v], axis=2)        # [1, T, 12]

np.savez("io.npz", qkv=qkv, output=output)
print(f"Saved io.npz  qkv={qkv.shape} output={output.shape}")
