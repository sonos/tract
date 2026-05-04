#!/usr/bin/env python3
"""Generate io.npz for max-pool then block-diag SDPA.

Pre-pool the queries via a 3-tap max pool (kernel=3, padding=1) on the
streaming axis, then block-diag SDPA(q_pooled, k, v) with scale=0.5.

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

q = rng.standard_normal((T, D)).astype(np.float32)
k = rng.standard_normal((T, D)).astype(np.float32)
v = rng.standard_normal((T, D)).astype(np.float32)

# Max-pool kernel=5, padding=2, stride=1 along the time axis.
# Delay = (kernel - 1) / 2 = 2, exactly one chunk at P=2.  tract's
# max_pool with border='constant' ignores padded positions
# (equivalent to filling with -inf in the max).
q_padded = np.pad(q, ((2, 2), (0, 0)), mode="constant", constant_values=-np.inf)
q_pre = np.stack(
    [q_padded[i : i + T] for i in range(5)], axis=0
).max(axis=0).astype(np.float32)

scores = np.einsum("id,jd->ij", q_pre, k)
idx = np.arange(T)
chunk_id = idx // P
in_block = (chunk_id[:, None] == chunk_id[None, :])
scaled = scores * SCALE
pre = np.where(in_block, scaled, -np.inf)
attn = np.exp(pre - pre.max(axis=-1, keepdims=True))
attn = (attn / attn.sum(axis=-1, keepdims=True)).astype(np.float32)
output = np.einsum("ij,jd->id", attn, v).astype(np.float32)

np.savez("io.npz", q=q, k=k, v=v, output=output)
print(f"Saved io.npz  q={q.shape} k={k.shape} v={v.shape} output={output.shape}")
