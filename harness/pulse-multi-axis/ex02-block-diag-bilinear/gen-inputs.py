#!/usr/bin/env python3
"""Generate io.npz for the block-diagonal-bilinear synthetic.

Q·Kᵀ → block-diagonal mask → attn·V, no softmax.

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

rng = np.random.default_rng(42)

a = rng.standard_normal((T, D)).astype(np.float32)
b = rng.standard_normal((T, D)).astype(np.float32)
c = rng.standard_normal((T, D)).astype(np.float32)

scores = np.einsum("id,jd->ij", a, b)                   # [T, T]
idx       = np.arange(T)
chunk_id  = idx // P
mask      = (chunk_id[:, None] == chunk_id[None, :]).astype(np.float32)
masked    = scores * mask
output    = np.einsum("ij,jd->id", masked, c).astype(np.float32)  # [T, D]

np.savez("io.npz", a=a, b=b, c=c, output=output)
print(f"Saved io.npz  a={a.shape} b={b.shape} c={c.shape} output={output.shape}")
