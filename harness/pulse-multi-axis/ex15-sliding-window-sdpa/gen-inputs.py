#!/usr/bin/env python3
"""Generate io.npz for the sliding-window SDPA synthetic.

Q·Kᵀ → ScaledMaskedSoftmax(scale=0.5, `0 <= i - j < W`) → attn·V, a
per-position causal window of W = 5 positions.

T = 12 covers the window filling up (i < 4) and sliding past it.
"""

import numpy as np

T, D, W = 12, 4, 5
SCALE = 0.5

rng = np.random.default_rng(42)

a = rng.standard_normal((T, D)).astype(np.float32)
b = rng.standard_normal((T, D)).astype(np.float32)
c = rng.standard_normal((T, D)).astype(np.float32)

scores = np.einsum("id,jd->ij", a, b)
idx = np.arange(T)
diff = idx[:, None] - idx[None, :]
mask = (diff >= 0) & (diff < W)

pre = np.where(mask, scores * SCALE, -np.inf)
attn = np.exp(pre - pre.max(axis=-1, keepdims=True))
attn = (attn / attn.sum(axis=-1, keepdims=True)).astype(np.float32)

output = np.einsum("ij,jd->id", attn, c).astype(np.float32)

np.savez("io.npz", a=a, b=b, c=c, output=output)
print(f"Saved io.npz  a={a.shape} b={b.shape} c={c.shape} output={output.shape}")
