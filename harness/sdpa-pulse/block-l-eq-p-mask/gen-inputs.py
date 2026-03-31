#!/usr/bin/env python3
"""Generate io.npz for the block-l-eq-p-mask test.

Parameters
----------
C=4 (chunks), P=2 (chunk/pulse size), Dh=8

Input:  qkv  [C, 3*P, Dh]  (axis 0 streams)
        mask [C, P,   P]   (bool, all-true — every token attends to all others in the chunk)
Output: [C, P, Dh]

The mask is all-true so the output is identical to block-l-eq-p.
The mask input is included to exercise the Iff + softmax pipeline.
"""

import numpy as np

C, P, Dh = 4, 2, 8

rng = np.random.default_rng(42)

q = rng.standard_normal((C, P, Dh)).astype(np.float32)
k = rng.standard_normal((C, P, Dh)).astype(np.float32)
v = rng.standard_normal((C, P, Dh)).astype(np.float32)

# All-true block-diagonal mask: every token attends to every other in the chunk
mask = np.ones((C, P, P), dtype=bool)

# scores[c, p, q] = Q[c,p,:] · K[c,q,:]
scores = np.einsum("cpd,cqd->cpq", q, k)               # [C, P, P]
# Boolean mask: keep scores where mask=True, -inf where False
fill = np.full_like(scores, -np.inf)
masked_scores = np.where(mask, scores, fill)
exp_s  = np.exp(masked_scores - masked_scores.max(axis=-1, keepdims=True))
attn   = exp_s / exp_s.sum(axis=-1, keepdims=True)     # [C, P, P]
output = np.einsum("cpq,cqd->cpd", attn, v).astype(np.float32)

qkv = np.concatenate([q, k, v], axis=1)

np.savez("io.npz", qkv=qkv, mask=mask, output=output)
print(f"Saved io.npz  qkv={qkv.shape} mask={mask.shape} output={output.shape}")
