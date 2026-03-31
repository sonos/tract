#!/usr/bin/env python3
"""Generate io.npz for the block-left-1-mask test.

Parameters
----------
T=8 (total tokens = C*P = 4 chunks × 2 tokens/chunk)
Dh=4, P=2 (chunk size), left_chunks=1

Input:  qkv [T, 3*Dh] = [8, 12]   (axis 0 streams at pulse_size=P=2)
Output: [T, Dh] = [8, 4]

Attention: token i attends to token j iff
    0 <= floor(i/P) - floor(j/P) <= left_chunks

No padding mask — all T tokens are valid.
"""

import numpy as np

T, Dh, P, left_chunks = 8, 4, 2, 1

rng = np.random.default_rng(42)

q = rng.standard_normal((T, Dh)).astype(np.float32)
k = rng.standard_normal((T, Dh)).astype(np.float32)
v = rng.standard_normal((T, Dh)).astype(np.float32)

# chunk index for each token position
chunk_idx = np.arange(T) // P          # [T]  int

# mask[i,j] = True iff 0 <= chunk_idx[i] - chunk_idx[j] <= left_chunks
diff = chunk_idx[:, None] - chunk_idx[None, :]   # [T, T]  int
mask = (diff >= 0) & (diff <= left_chunks)        # [T, T]  bool

# sanity: print the mask for inspection
print("Mask (T=8, P=2, left_chunks=1):")
for row in mask:
    print("  ", "".join("1" if x else "0" for x in row))

# attention with mask
scores      = q @ k.T                                   # [T, T]
fill        = np.full_like(scores, -np.inf)
masked_scores = np.where(mask, scores, fill)
# stable softmax per query row
mx          = masked_scores.max(axis=1, keepdims=True)
exp_s       = np.exp(masked_scores - mx)
attn        = exp_s / exp_s.sum(axis=1, keepdims=True)  # [T, T]
output      = (attn @ v).astype(np.float32)             # [T, Dh]

qkv = np.concatenate([q, k, v], axis=1)                # [T, 3*Dh]

np.savez("io.npz", qkv=qkv, output=output)
print(f"Saved io.npz  qkv={qkv.shape} output={output.shape}")
