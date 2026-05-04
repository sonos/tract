#!/usr/bin/env python3
"""Generate io.npz for the block-diag select+softmax SDPA synthetic.

Same semantics as ex06 but with explicit `select(mask, scores, -inf)`
instead of `ScaledMaskedSoftmax` — see graph.nnef for why.  No scale.

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

scores = np.einsum("id,jd->ij", a, b)              # [T, T]

idx       = np.arange(T)
chunk_id  = idx // P
in_block  = (chunk_id[:, None] == chunk_id[None, :])  # bool [T, T]

pre       = np.where(in_block, scores, -np.inf)
attn      = np.exp(pre - pre.max(axis=-1, keepdims=True))
attn      = attn / attn.sum(axis=-1, keepdims=True)
attn      = attn.astype(np.float32)

output    = np.einsum("ij,jd->id", attn, c).astype(np.float32)  # [T, D]

np.savez("io.npz", a=a, b=b, c=c, output=output)
print(f"Saved io.npz  a={a.shape} b={b.shape} c={c.shape} output={output.shape}")
