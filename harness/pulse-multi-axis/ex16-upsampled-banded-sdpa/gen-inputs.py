#!/usr/bin/env python3
"""Generate io.npz for the upsampled banded SDPA synthetic.

x [1, 4, F] is repeated 4x along time (the graph's all-ones stride-4
depthwise deconv), giving t [4F, 4].  Then t·tᵀ → ScaledMaskedSoftmax
(scale=0.5, `0 <= chunk(i) - chunk(j) <= 1`, chunks of 2) → attn·t.

F = 3 frames, so T = 12 positions and 6 mask chunks.
"""

import numpy as np

F, D, UP, K, L = 3, 4, 4, 2, 1
SCALE = 0.5

rng = np.random.default_rng(42)

x = rng.standard_normal((1, D, F)).astype(np.float32)
t = np.repeat(x[0].T, UP, axis=0)

scores = t @ t.T
chunk_id = np.arange(F * UP) // K
diff = chunk_id[:, None] - chunk_id[None, :]
mask = (diff >= 0) & (diff <= L)

pre = np.where(mask, scores * SCALE, -np.inf)
attn = np.exp(pre - pre.max(axis=-1, keepdims=True))
attn = (attn / attn.sum(axis=-1, keepdims=True)).astype(np.float32)

output = (attn @ t).astype(np.float32)

np.savez("io.npz", x=x, output=output)
print(f"Saved io.npz  x={x.shape} output={output.shape}")
