#!/usr/bin/env python3
"""Generate io.npz for the block-attention left_chunks=1 test.

Parameters
----------
C=4 (chunks), P=2 (chunk/pulse size), Dh=8
T = C * P = 8 tokens total.

Input:  qkv  [C, 3*P, Dh]  (axis 0 streams, axis 1 = Q/K/V)
Output: [C, P, Dh]

Each chunk c attends over concat(K[c-1], K[c]) and V[c-1], V[c]).
K[c-1] = 0 for c=0 (delay buffer initialised to zero).
"""

import numpy as np

C, P, Dh = 4, 2, 8

rng = np.random.default_rng(42)

q = rng.standard_normal((C, P, Dh)).astype(np.float32)   # [C, P, Dh]
k = rng.standard_normal((C, P, Dh)).astype(np.float32)
v = rng.standard_normal((C, P, Dh)).astype(np.float32)

# Previous-chunk K and V (zero-padded at c=0)
k_prev = np.concatenate([np.zeros((1, P, Dh), dtype=np.float32), k[:-1]], axis=0)
v_prev = np.concatenate([np.zeros((1, P, Dh), dtype=np.float32), v[:-1]], axis=0)

# Concatenate previous + current on the token axis  [C, 2P, Dh]
k_ctx = np.concatenate([k_prev, k], axis=1)
v_ctx = np.concatenate([v_prev, v], axis=1)

# scores[c, p, l] = Q[c,p,:] · K_ctx[c,l,:]
scores = np.einsum("cpd,cld->cpl", q, k_ctx)               # [C, P, 2P]
exp_s  = np.exp(scores - scores.max(axis=-1, keepdims=True))
attn   = exp_s / exp_s.sum(axis=-1, keepdims=True)          # [C, P, 2P]
output = np.einsum("cpl,cld->cpd", attn, v_ctx).astype(np.float32)  # [C, P, Dh]

# Pack Q/K/V flat along axis 1 → [C, 3*P, Dh] = [C, 6, 8]
qkv = np.concatenate([q, k, v], axis=1)

np.savez("io.npz", qkv=qkv, output=output)
print(f"Saved io.npz  qkv={qkv.shape} output={output.shape}")
