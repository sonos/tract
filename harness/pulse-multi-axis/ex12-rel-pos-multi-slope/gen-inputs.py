#!/usr/bin/env python3
"""ex12-rel-pos-multi-slope — io.npz generator.

Block-diagonal SDPA + Transformer-XL relative position bias.  T = 4,
chunk size P = 2 → 2 chunks of 2 tokens each.

Position scores use the skew trick:
  pos_raw   = a · posᵀ          # [T, 2T-1] = [4, 7]
  pos_view  = pad-and-reshape   # the standard skew sequence
  pos_sliced + reshape + slice  → pos_scores [T, T]

Final attention is per-chunk (block-diagonal mask) → matmul(attn, c).
"""
import numpy as np

T, D, P = 4, 4, 2
rng = np.random.default_rng(12)
a = rng.standard_normal((T, D)).astype(np.float32)
b = rng.standard_normal((T, D)).astype(np.float32)
c = rng.standard_normal((T, D)).astype(np.float32)
pos = rng.standard_normal((2 * T - 1, D)).astype(np.float32)


def skew(pos_raw: np.ndarray) -> np.ndarray:
    """[T, 2T-1] -> [T, T] via standard Transformer-XL skew trick."""
    T_in = pos_raw.shape[0]
    padded = np.concatenate([np.zeros((T_in, 1), dtype=pos_raw.dtype), pos_raw], axis=1)
    view = padded.reshape(-1, T_in)               # [2T, T]
    sliced = view[1 : 2 * T_in, :]                # [2T-1, T]
    bd = sliced.reshape(T_in, -1)                 # [T, 2T-1]
    return bd[:, :T_in]                           # [T, T]


content = a @ b.T                                 # [T, T]
pos_raw = a @ pos.T                               # [T, 2T-1]
pos_scores = skew(pos_raw)                        # [T, T]
scores = (content + pos_scores) * 0.5             # apply scale upfront

idx = np.arange(T)
chunk = idx // P
in_block = (chunk[:, None] == chunk[None, :])
masked = np.where(in_block, scores, -np.inf)

m = masked.max(axis=-1, keepdims=True)
exp_s = np.exp(masked - m)
attn = exp_s / exp_s.sum(axis=-1, keepdims=True)

output = (attn @ c).astype(np.float32)

np.savez("io.npz", a=a, b=b, c=c, pos=pos, output=output)
print(f"Saved io.npz  a={a.shape} pos={pos.shape} output={output.shape}")
