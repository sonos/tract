#!/usr/bin/env python3
"""Generate io.npz for the ex06-batch-multihead test.

Parameters
----------
B=1, H=2, T=8 (C*P = 4 chunks × 2 tokens/chunk), Dh=4, P=2, left_chunks=1

Input:  qkv [1, 2, T, 3*Dh] = [1, 2, 8, 12]   (axis 2 streams at pulse_size=P=2)
Output: [1, 2, T, Dh] = [1, 2, 8, 4]

Reference uses the flat T×T masked attention with -inf for out-of-window tokens,
matching the unoptimised batch graph.  The mask [T,T] is broadcast to [1,1,T,T].
"""

import numpy as np

B, H, T, Dh, P, left_chunks = 1, 2, 8, 4, 2, 1

rng = np.random.default_rng(42)

q = rng.standard_normal((B, H, T, Dh)).astype(np.float32)  # [1, 2, 8, 4]
k = rng.standard_normal((B, H, T, Dh)).astype(np.float32)
v = rng.standard_normal((B, H, T, Dh)).astype(np.float32)

# Chunk mask [T, T]
chunk_idx = np.arange(T) // P
diff  = chunk_idx[:, None] - chunk_idx[None, :]   # [T, T]
mask  = (diff >= 0) & (diff <= left_chunks)

print("Mask (T=8, P=2, left_chunks=1):")
for row in mask:
    print("  ", "".join("1" if x else "0" for x in row))

# scores [B, H, T, T]
scores = np.einsum('bhtd,bhsd->bhts', q, k)
masked = np.where(mask[None, None, :, :], scores, -np.inf)

# softmax over key axis (axis 3)
mx    = masked.max(axis=3, keepdims=True)
exp_s = np.exp(masked - mx)
attn  = exp_s / exp_s.sum(axis=3, keepdims=True)

output = np.einsum('bhts,bhsd->bhtd', attn, v).astype(np.float32)  # [1, 2, 8, 4]
qkv    = np.concatenate([q, k, v], axis=-1)                         # [1, 2, 8, 12]

np.savez("io.npz", qkv=qkv, output=output)
print(f"Saved io.npz  qkv={qkv.shape} output={output.shape}")
