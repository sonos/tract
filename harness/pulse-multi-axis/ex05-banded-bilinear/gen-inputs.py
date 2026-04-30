#!/usr/bin/env python3
"""Generate io.npz for the banded-bilinear synthetic.

ex02-style Q·Kᵀ → mask → attn·V structure with ex03's banded mask.
Mask `0 ≤ chunk(i) - chunk(j) ≤ 1` combined with the bilinear's
contraction over j gives, per output i-chunk c_i, a sum over j-chunks
in [c_i - 1, c_i] — past+current.  Streaming output is causal in i.

Parameters
----------
P = 2 (pulse / chunk size)
S = 3 (chunks)
T = S * P = 6
D = 4
L = 1 (band width on diff: 0..L)
"""

import numpy as np

P, S, D, L = 2, 3, 4, 1
T = S * P

rng = np.random.default_rng(42)

a = rng.standard_normal((T, D)).astype(np.float32)
b = rng.standard_normal((T, D)).astype(np.float32)
c = rng.standard_normal((T, D)).astype(np.float32)

scores = np.einsum("id,jd->ij", a, b)                   # [T, T]
idx       = np.arange(T)
chunk_id  = idx // P
diff      = chunk_id[:, None] - chunk_id[None, :]
mask      = ((diff >= 0) & (diff <= L)).astype(np.float32)
masked    = scores * mask
output    = np.einsum("ij,jd->id", masked, c).astype(np.float32)  # [T, D]

np.savez("io.npz", a=a, b=b, c=c, output=output)
print(f"Saved io.npz  a={a.shape} b={b.shape} c={c.shape} output={output.shape}")
