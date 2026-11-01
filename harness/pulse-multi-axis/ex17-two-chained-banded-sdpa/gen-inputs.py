#!/usr/bin/env python3
"""Generate io.npz for two chained banded SDPA blocks.

Block 1: SDPA(q,      k1, v1), block 2: SDPA(block1, k2, v2), both with the
mask `0 <= chunk(i) - chunk(j) <= 1` over chunks of 2, scale 0.5.

T = 8 positions, 4 chunks.
"""

import numpy as np

P, S, D, L = 2, 4, 4, 1
T = S * P
SCALE = 0.5

rng = np.random.default_rng(42)

q = rng.standard_normal((T, D)).astype(np.float32)
k1 = rng.standard_normal((T, D)).astype(np.float32)
v1 = rng.standard_normal((T, D)).astype(np.float32)
k2 = rng.standard_normal((T, D)).astype(np.float32)
v2 = rng.standard_normal((T, D)).astype(np.float32)

chunk_id = np.arange(T) // P
diff = chunk_id[:, None] - chunk_id[None, :]
in_band = (diff >= 0) & (diff <= L)


def banded_sdpa(qx, kx, vx):
    pre = np.where(in_band, np.einsum("id,jd->ij", qx, kx) * SCALE, -np.inf)
    attn = np.exp(pre - pre.max(axis=-1, keepdims=True))
    attn = (attn / attn.sum(axis=-1, keepdims=True)).astype(np.float32)
    return np.einsum("ij,jd->id", attn, vx).astype(np.float32)


block1 = banded_sdpa(q, k1, v1)
output = banded_sdpa(block1, k2, v2)

np.savez("io.npz", q=q, k1=k1, v1=v1, k2=k2, v2=v2, output=output)
print(f"Saved io.npz  output={output.shape}")
