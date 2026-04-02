#!/usr/bin/env python3
"""Generate io.npz for the ex07-block-left-1-chunkpos test.

Parameters
----------
T=8 (total tokens = C*P = 4 chunks × 2 tokens/chunk)
Dh=4, P=2 (chunk size), left_chunks=1, slope=-0.5

Input:  qkv [T, 3*Dh] = [8, 12]   (axis 0 streams at pulse_size=P=2)
Output: [T, Dh] = [8, 4]

Position bias: v_bias[i,j] = slope * (floor(i/P) - floor(j/P))
             = -0.5 * (chunk_idx[i] - chunk_idx[j])

This is the Transformer-XL v-bias concept (constant additive term that
depends only on the chunk-index difference, not on Q or K values).

At pulse time the binary pulsifier materialises chunk_diff as a constant
[P, key_window] tensor by evaluating Div() in the TDim coordinate
expression at steady-state coordinates.

Reference uses flat T×T masked attention with -inf for out-of-window tokens.
"""

import numpy as np

T, Dh, P, left_chunks = 8, 4, 2, 1
slope = -0.5

rng = np.random.default_rng(42)

q = rng.standard_normal((T, Dh)).astype(np.float32)
k = rng.standard_normal((T, Dh)).astype(np.float32)
v = rng.standard_normal((T, Dh)).astype(np.float32)

i_idx = np.arange(T)
j_idx = np.arange(T)

# Chunk mask
chunk_idx = i_idx // P
diff  = chunk_idx[:, None] - chunk_idx[None, :]   # [T, T]
mask  = (diff >= 0) & (diff <= left_chunks)

# Chunk-level position bias: slope * (chunk_idx[i] - chunk_idx[j])
chunk_diff = chunk_idx[:, None] - chunk_idx[None, :]   # [T, T] i64
pos_bias   = (slope * chunk_diff).astype(np.float32)

print("Chunk-level position bias (T=8, P=2, left_chunks=1, slope=-0.5):")
for row in pos_bias:
    print("  ", " ".join(f"{x:+.1f}" for x in row))

scores = q @ k.T + pos_bias                    # [T, T]
masked = np.where(mask, scores, -np.inf)

mx    = masked.max(axis=1, keepdims=True)
exp_s = np.exp(masked - mx)
attn  = exp_s / exp_s.sum(axis=1, keepdims=True)

output = (attn @ v).astype(np.float32)         # [T, Dh]
qkv    = np.concatenate([q, k, v], axis=1)     # [T, 3*Dh]

np.savez("io.npz", qkv=qkv, output=output)
print(f"Saved io.npz  qkv={qkv.shape} output={output.shape}")
