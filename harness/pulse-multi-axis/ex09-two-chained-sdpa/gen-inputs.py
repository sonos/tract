#!/usr/bin/env python3
"""Generate io.npz for two chained block-diag SDPA layers.

Layer 1: SDPA(q,  k1, v1)        with block-diag mask, scale=0.5.
Layer 2: SDPA(L1, k2, v2)        same mask, scale=0.5.

Parameters
----------
P = 2 (pulse / chunk size)
S = 3 (chunks)
T = S * P = 6
D = 4
"""

import numpy as np

P, S, D = 2, 3, 4
T = S * P
SCALE = 0.5

rng = np.random.default_rng(42)

q  = rng.standard_normal((T, D)).astype(np.float32)
k1 = rng.standard_normal((T, D)).astype(np.float32)
v1 = rng.standard_normal((T, D)).astype(np.float32)
k2 = rng.standard_normal((T, D)).astype(np.float32)
v2 = rng.standard_normal((T, D)).astype(np.float32)


def block_diag_sdpa(qx, kx, vx):
    scores = np.einsum("id,jd->ij", qx, kx)
    idx = np.arange(T)
    chunk_id = idx // P
    in_block = (chunk_id[:, None] == chunk_id[None, :])
    scaled = scores * SCALE
    pre = np.where(in_block, scaled, -np.inf)
    attn = np.exp(pre - pre.max(axis=-1, keepdims=True))
    attn = attn / attn.sum(axis=-1, keepdims=True)
    return np.einsum("ij,jd->id", attn.astype(np.float32), vx).astype(np.float32)


layer1 = block_diag_sdpa(q, k1, v1)
output = block_diag_sdpa(layer1, k2, v2)

np.savez("io.npz", q=q, k1=k1, v1=v1, k2=k2, v2=v2, output=output)
print(f"Saved io.npz  q={q.shape} k1={k1.shape} v1={v1.shape} k2={k2.shape} v2={v2.shape} output={output.shape}")
