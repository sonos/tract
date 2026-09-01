#!/usr/bin/env python3
"""Generate io.npz for the batched banded-reduce synthetic.

Same banded geometry as ex03 (every row at chunk c interacts with chunks
{c, c-1, ..., c-L}), with a batch axis on the data and none on the mask.

Parameters
----------
B = 2 (batch)
P = 2 (pulse / chunk size)
C = 3 (chunks)
T = C * P = 6 (total stream length)
D = 4 (per-token feature dim)
L = 1 (left-context, in chunks)
"""

import numpy as np

B, P, C, D, L = 2, 2, 3, 4, 1
T = C * P

rng = np.random.default_rng(42)

a = rng.standard_normal((B, T, D)).astype(np.float32)
b = rng.standard_normal((B, T, D)).astype(np.float32)

scores = np.einsum("nid,njd->nij", a, b)                # [B, T, T]

idx      = np.arange(T)
chunk_id = idx // P
diff     = chunk_id[:, None] - chunk_id[None, :]
mask     = ((diff >= 0) & (diff <= L)).astype(np.float32)

masked = scores * mask[None, :, :]
output = masked.sum(axis=1).astype(np.float32)          # [B, T]

np.savez("io.npz", a=a, b=b, output=output)
print(f"Saved io.npz  a={a.shape} b={b.shape} output={output.shape}")
