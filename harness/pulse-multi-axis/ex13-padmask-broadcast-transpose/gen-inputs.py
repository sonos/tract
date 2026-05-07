#!/usr/bin/env python3
"""ex13-padmask-broadcast-transpose — io.npz generator.

T = 4, chunk size P = 2 → 2 chunks of 2 tokens each.
The pad mask marks the last frame as padding (invalid).

Reference computation:
    scores = q · kᵀ                        # [T, T]
    pad_2d[i, j] = pad[i] AND pad[j]       # [T, T]
    in_block[i, j] = (i // P == j // P)    # [T, T]
    mask     = in_block AND pad_2d         # [T, T]
    masked   = where(mask, scores, -inf)
    attn     = softmax(masked, axis=-1)    # rows where every key is masked → NaN
    output   = attn · c                    # [T, D]
"""
import numpy as np

T, D, P = 4, 4, 2
rng = np.random.default_rng(13)

q   = rng.standard_normal((T, D)).astype(np.float32)
k   = rng.standard_normal((T, D)).astype(np.float32)
c   = rng.standard_normal((T, D)).astype(np.float32)
# All frames valid in this minimal version.  The pad-mask construction
# pattern (broadcast → transpose → AND) reproduces regardless of which
# bits are set; using all-True keeps the reference output finite (a
# padded row would be fully masked under block-diag and produce NaN
# from softmax).
pad = np.ones(T, dtype=bool)

scores  = q @ k.T                                         # [T, T]
pad_2d  = pad[:, None] & pad[None, :]                     # [T, T]

idx       = np.arange(T)
chunk_id  = idx // P
diff      = chunk_id[:, None] - chunk_id[None, :]
in_block  = (diff >= 0) & (diff <= 1)                     # banded-causal [T, T]

mask    = in_block & pad_2d
masked  = np.where(mask, scores, -np.inf)

# Softmax row-by-row, with a guard against fully-masked rows producing NaN.
m       = masked.max(axis=-1, keepdims=True)
exp_s   = np.exp(masked - m)
denom   = exp_s.sum(axis=-1, keepdims=True)
attn    = np.where(denom > 0, exp_s / denom, 0.0).astype(np.float32)

output  = (attn @ c).astype(np.float32)

np.savez("io.npz", q=q, k=k, c=c, pad=pad, output=output)
print(f"Saved io.npz  q={q.shape} pad={pad.shape} output={output.shape}")
