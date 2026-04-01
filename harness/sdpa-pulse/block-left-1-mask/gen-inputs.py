#!/usr/bin/env python3
"""Generate io.npz for the block-left-1-mask test.

Parameters
----------
T=8 (total tokens = C*P = 4 chunks × 2 tokens/chunk)
Dh=4, P=2 (chunk size), left_chunks=1

Input:  qkv [T, 3*Dh] = [8, 12]   (axis 0 streams at pulse_size=P=2)
Output: [T, Dh] = [8, 4]

Attention: computed per chunk using zero-padded K/V context window.
For chunk c: k_ctx[c] = concat([k[c-1] or zeros, k[c]], axis=0)  [2P, Dh]
This matches the FoldWindowAttention transformation applied by the optimizer,
which replaces the T×T masked attention with bounded-window chunk attention.
Positions from before the sequence start are zero-padded (score=0, not -inf).
"""

import numpy as np

T, Dh, P, left_chunks = 8, 4, 2, 1
C = T // P  # number of chunks

rng = np.random.default_rng(42)

q = rng.standard_normal((T, Dh)).astype(np.float32)
k = rng.standard_normal((T, Dh)).astype(np.float32)
v = rng.standard_normal((T, Dh)).astype(np.float32)

# show the logical mask for reference
chunk_idx = np.arange(T) // P
diff = chunk_idx[:, None] - chunk_idx[None, :]
mask = (diff >= 0) & (diff <= left_chunks)
print("Mask (T=8, P=2, left_chunks=1):")
for row in mask:
    print("  ", "".join("1" if x else "0" for x in row))

# Compute attention in chunk layout with zero-padded K/V context.
# This matches FoldWindowAttention: no Iff masking; startup positions are zero-padded.
q_c = q.reshape(C, P, Dh)   # [C, P, Dh]
k_c = k.reshape(C, P, Dh)   # [C, P, Dh]
v_c = v.reshape(C, P, Dh)   # [C, P, Dh]

output_chunks = []
for c in range(C):
    # Build K and V context: [2P, Dh]
    k_parts = []
    v_parts = []
    for lag in range(left_chunks, 0, -1):
        prev = c - lag
        if prev < 0:
            k_parts.append(np.zeros((P, Dh), dtype=np.float32))
            v_parts.append(np.zeros((P, Dh), dtype=np.float32))
        else:
            k_parts.append(k_c[prev])
            v_parts.append(v_c[prev])
    k_parts.append(k_c[c])
    v_parts.append(v_c[c])
    k_ctx = np.concatenate(k_parts, axis=0)  # [(L+1)*P, Dh]
    v_ctx = np.concatenate(v_parts, axis=0)

    scores = q_c[c] @ k_ctx.T   # [P, (L+1)*P]
    # stable softmax over context axis
    mx = scores.max(axis=1, keepdims=True)
    exp_s = np.exp(scores - mx)
    attn = exp_s / exp_s.sum(axis=1, keepdims=True)
    output_chunks.append(attn @ v_ctx)   # [P, Dh]

output = np.concatenate(output_chunks, axis=0).astype(np.float32)  # [T, Dh]
qkv = np.concatenate([q, k, v], axis=1)                            # [T, 3*Dh]

np.savez("io.npz", qkv=qkv, output=output)
print(f"Saved io.npz  qkv={qkv.shape} output={output.shape}")
