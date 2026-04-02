#!/usr/bin/env python3
"""Generate io.npz for the ex04-block-left-1-mask test.

Parameters
----------
T=8 (total tokens = C*P = 4 chunks × 2 tokens/chunk)
Dh=4, P=2 (chunk size), left_chunks=1

Input:  qkv [T, 3*Dh] = [8, 12]   (axis 0 streams at pulse_size=P=2)
Output: [T, Dh] = [8, 4]

Reference uses the flat T×T masked attention matching the unoptimised batch
graph: Iff masking with -inf for out-of-window tokens.

In streaming, the pulsifier uses ChunkWindowMask to handle the windowed
attention. Intermediate pulsed shapes ([P, key_window]) differ from the
reference ([S, S]) but the final output matches; compare --stream skips
incompatible-shape intermediates rather than failing on them.
"""

import numpy as np

T, Dh, P, left_chunks = 8, 4, 2, 1

rng = np.random.default_rng(42)

q = rng.standard_normal((T, Dh)).astype(np.float32)
k = rng.standard_normal((T, Dh)).astype(np.float32)
v = rng.standard_normal((T, Dh)).astype(np.float32)

# Full T×T attention with chunk mask (-inf for out-of-window).
chunk_idx = np.arange(T) // P
diff = chunk_idx[:, None] - chunk_idx[None, :]   # [T, T]
mask = (diff >= 0) & (diff <= left_chunks)

print("Mask (T=8, P=2, left_chunks=1):")
for row in mask:
    print("  ", "".join("1" if x else "0" for x in row))

scores = q @ k.T                                  # [T, T]
masked = np.where(mask, scores, -np.inf)

# stable softmax over axis 1
mx    = masked.max(axis=1, keepdims=True)
exp_s = np.exp(masked - mx)
attn  = exp_s / exp_s.sum(axis=1, keepdims=True)

output = (attn @ v).astype(np.float32)            # [T, Dh]
qkv    = np.concatenate([q, k, v], axis=1)        # [T, 3*Dh]

np.savez("io.npz", qkv=qkv, output=output)
print(f"Saved io.npz  qkv={qkv.shape} output={output.shape}")
