#!/usr/bin/env python3
"""Generate io.npz for the banded-reduce synthetic.

Banded mask geometry: every row at chunk c attends to chunks
{c, c-1, ..., c-L}.  For L = 1 and P = 2 this is a P-block lower
bidiagonal.

Parameters
----------
P = 2 (pulse / chunk size)
C = 3 (chunks)
T = C * P = 6 (total stream length)
D = 4 (per-token feature dim)
L = 1 (left-context, in chunks)
"""

import numpy as np

P, C, D, L = 2, 3, 4, 1
T = C * P

rng = np.random.default_rng(42)

a = rng.standard_normal((T, D)).astype(np.float32)
b = rng.standard_normal((T, D)).astype(np.float32)

scores = np.einsum("id,jd->ij", a, b)                   # [T, T]

idx       = np.arange(T)
chunk_id  = idx // P
diff      = chunk_id[:, None] - chunk_id[None, :]
mask      = ((diff >= 0) & (diff <= L)).astype(np.float32)

masked = scores * mask
output = masked.sum(axis=0).astype(np.float32)          # [T]

np.savez("io.npz", a=a, b=b, output=output)
print(f"Saved io.npz  a={a.shape} b={b.shape} output={output.shape}")
