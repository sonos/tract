#!/usr/bin/env python3
"""Generate io.npz for the banded causal SDPA synthetic.

Q·Kᵀ → ScaledMaskedSoftmax(scale=0.5, banded bool mask) → attn·V, with
mask `0 ≤ chunk(i) - chunk(j) ≤ 1` (causal in i with 1-chunk left
context).

Per output i-chunk c_i, attention runs over j with chunk(j) ∈ [c_i - 1,
c_i].  At c_i = 0 the past j-chunk doesn't exist; the softmax shrinks
to chunk 0's k positions only.

Parameters
----------
P = 2 (pulse / chunk size)
S = 3 (chunks)
T = S * P = 6
D = 4
L = 1 (band width: 0..L)
"""

import numpy as np

P, S, D, L = 2, 3, 4, 1
T = S * P
SCALE = 0.5

rng = np.random.default_rng(42)

a = rng.standard_normal((T, D)).astype(np.float32)
b = rng.standard_normal((T, D)).astype(np.float32)
c = rng.standard_normal((T, D)).astype(np.float32)

scores = np.einsum("id,jd->ij", a, b)              # [T, T]
idx       = np.arange(T)
chunk_id  = idx // P
diff      = chunk_id[:, None] - chunk_id[None, :]
mask      = (diff >= 0) & (diff <= L)              # bool [T, T]

scaled    = scores * SCALE
pre       = np.where(mask, scaled, -np.inf)
attn      = np.exp(pre - pre.max(axis=-1, keepdims=True))
attn      = attn / attn.sum(axis=-1, keepdims=True)
attn      = attn.astype(np.float32)

output    = np.einsum("ij,jd->id", attn, c).astype(np.float32)  # [T, D]

np.savez("io.npz", a=a, b=b, c=c, output=output)
print(f"Saved io.npz  a={a.shape} b={b.shape} c={c.shape} output={output.shape}")
