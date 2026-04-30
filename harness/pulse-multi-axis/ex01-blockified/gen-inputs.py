#!/usr/bin/env python3
"""Generate io.npz for the blockified (post-rewrite) form, with the
original [T] model interface preserved.

Reuses the same numerical inputs as the parent ex01-block-diag-reduce
(seed=42), and the same output shape [T] — the chunking happens inside
the graph, not at the boundary.
"""

import numpy as np

P, S, D = 2, 3, 4
T = S * P

rng = np.random.default_rng(42)

a = rng.standard_normal((T, D)).astype(np.float32)
b = rng.standard_normal((T, D)).astype(np.float32)

a_blk = a.reshape(S, P, D)
b_blk = b.reshape(S, P, D)

block_scores = np.einsum("spd,sqd->spq", a_blk, b_blk)         # [S, P, P]
output       = block_scores.sum(axis=1).reshape(T).astype(np.float32)  # [T]

np.savez("io.npz", a=a, b=b, output=output)
print(f"Saved io.npz  a={a.shape} b={b.shape} output={output.shape}")
