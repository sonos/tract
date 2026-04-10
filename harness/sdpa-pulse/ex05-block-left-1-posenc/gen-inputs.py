#!/usr/bin/env python3
"""Generate io.npz for the ex05-block-left-1-posenc test.

Parameters
----------
T=8 (total tokens = C*P = 4 chunks × 2 tokens/chunk)
Dh=4, P=2 (chunk size), left_chunks=1, slope=0.125

Input:  qkv [T, 3*Dh] = [8, 12]   (axis 0 streams at pulse_size=P=2)
Output: [T, Dh] = [8, 4]

Reference uses the flat T×T masked attention with pos_bias, matching the
unoptimised batch graph (Iff masking with -inf for out-of-window tokens).

In streaming, the pulsifier uses ChunkWindowMask + binary pulsifier to
handle the windowed attention.
Intermediate pulsed shapes ([P, key_window]) differ from the reference
([S, S]) but the final output matches; compare --stream skips incompatible-
shape intermediates rather than failing on them.
"""

import numpy as np

T, Dh, P, left_chunks = 8, 4, 2, 1
slope = 0.125

rng = np.random.default_rng(42)

q = rng.standard_normal((T, Dh)).astype(np.float32)
k = rng.standard_normal((T, Dh)).astype(np.float32)
v = rng.standard_normal((T, Dh)).astype(np.float32)

# Full T×T attention with pos_bias and chunk mask (-inf for out-of-window).
i_idx = np.arange(T)
j_idx = np.arange(T)

chunk_idx = i_idx // P
diff = chunk_idx[:, None] - chunk_idx[None, :]      # [T, T]
mask = (diff >= 0) & (diff <= left_chunks)

rel_pos  = i_idx[:, None] - j_idx[None, :]          # [T, T]: i - j
pos_bias = (-slope * rel_pos).astype(np.float32)

scores = q @ k.T + pos_bias                         # [T, T]
masked = np.where(mask, scores, -np.inf)

# stable softmax over axis 1
mx    = masked.max(axis=1, keepdims=True)
exp_s = np.exp(masked - mx)
attn  = exp_s / exp_s.sum(axis=1, keepdims=True)

output = (attn @ v).astype(np.float32)              # [T, Dh]
qkv    = np.concatenate([q, k, v], axis=1)          # [T, 3*Dh]

np.savez("io.npz", qkv=qkv, output=output)
print(f"Saved io.npz  qkv={qkv.shape} output={output.shape}")
