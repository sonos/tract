#!/usr/bin/env python3
"""Generate io.npz for ex09-batch-multihead-mask.

B=1, H=2, T=8, P=2, left_chunks=1, Dh=4
Input:  qkv [1, T, 24]   (Q|K|V each [1,T,H*Dh]=[1,T,8])
Output: [1, T, 8]
"""
import numpy as np

T, Dh, H, P, left_chunks, B = 8, 4, 2, 2, 1, 1
rng = np.random.default_rng(42)

q = rng.standard_normal((B, T, H, Dh)).astype(np.float32)
k = rng.standard_normal((B, T, H, Dh)).astype(np.float32)
v = rng.standard_normal((B, T, H, Dh)).astype(np.float32)

chunk_idx = np.arange(T) // P
diff = chunk_idx[:, None] - chunk_idx[None, :]   # [T, T]
mask = (diff >= 0) & (diff <= left_chunks)        # [T, T]

scores = np.einsum("bihd,bjhd->bhij", q, k)      # [B, H, T, T]
masked = np.where(mask[None, None], scores, -np.inf)

mx    = masked.max(axis=3, keepdims=True)
exp_s = np.exp(masked - mx)
attn  = exp_s / exp_s.sum(axis=3, keepdims=True)

ctx    = np.einsum("bhij,bjhd->bihd", attn, v)   # [B, T, H, Dh]
output = ctx.reshape(B, T, H * Dh).astype(np.float32)

q_flat = q.reshape(B, T, H * Dh)
k_flat = k.reshape(B, T, H * Dh)
v_flat = v.reshape(B, T, H * Dh)
qkv    = np.concatenate([q_flat, k_flat, v_flat], axis=2)  # [1, T, 24]

np.savez("io.npz", qkv=qkv, output=output)
print(f"Saved io.npz  qkv={qkv.shape} output={output.shape}")
