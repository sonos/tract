#!/usr/bin/env python3
"""Generate io.npz for the block-attention L=P test.

Parameters
----------
C=3 (chunks), P=2 (chunk/pulse size = L), Dh=8
T = C * P = 6 tokens total.

Input: qkv  [C, 3, P, Dh]  (axis 0 streams, axis 1 = Q/K/V)
Output: [C, P, Dh]
"""

import numpy as np

C, P, Dh = 3, 2, 8

rng = np.random.default_rng(42)

q = rng.standard_normal((C, P, Dh)).astype(np.float32)  # [C, P, Dh]
k = rng.standard_normal((C, P, Dh)).astype(np.float32)
v = rng.standard_normal((C, P, Dh)).astype(np.float32)

# Reference: block attention
# scores[c, p, q] = Q[c,p,:] · K[c,q,:]
scores = np.einsum("cpd,cqd->cpq", q, k)                # [C, P, P]
exp_s  = np.exp(scores - scores.max(axis=-1, keepdims=True))
attn   = exp_s / exp_s.sum(axis=-1, keepdims=True)      # [C, P, P]
output = np.einsum("cpq,cqd->cpd", attn, v).astype(np.float32)  # [C, P, Dh]

# Pack Q/K/V flat along axis 1 → [C, 3*P, Dh] = [C, 6, 8]
qkv = np.concatenate([q, k, v], axis=1)

np.savez("io.npz", qkv=qkv, output=output)
print(f"Saved io.npz  qkv={qkv.shape} output={output.shape}")
