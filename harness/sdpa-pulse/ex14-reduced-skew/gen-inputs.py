#!/usr/bin/env python3
"""Generate io.npz and r_pos.dat for ex14-reduced-skew.

Reduced version of ex14-rel-pos-skew-large-table: same skew trick with
fixed r_pos=[2*T-1, Dh]=[15, 4], but uses separate q/k/v inputs instead
of a combined qkv tensor.

Parameters: T=8, P=2, left_chunks=1, W=4, Dh=4, B=1
r_pos shape = [2*T-1, Dh] = [15, 4]
"""

import struct
import numpy as np

T, Dh, P, left_chunks, B = 8, 4, 2, 1, 1
W = (left_chunks + 1) * P   # 4
R = 2 * T - 1               # 15

rng = np.random.default_rng(42)

q     = rng.standard_normal((B, T, Dh)).astype(np.float32)
k     = rng.standard_normal((B, T, Dh)).astype(np.float32)
v     = rng.standard_normal((B, T, Dh)).astype(np.float32)
r_pos = rng.standard_normal((R, Dh)).astype(np.float32)  # [15, 4]

content_scores = np.einsum("bid,bjd->bij", q, k)   # [1, T, T]
pos_raw        = np.einsum("bid,jd->bij",  q, r_pos)  # [1, T, 15]

# Skew trick
pos_padded = np.pad(pos_raw, [[0,0],[0,0],[1,0]])   # [1, T, 16]
pos_view   = pos_padded.reshape(B, 2*T, T)           # [1, 16, T]
pos_sliced = pos_view[:, 1:2*T, :]                   # [1, 2T-1=15, T]
pos_bd     = pos_sliced.reshape(B, T, 2*T-1)         # [1, T, 15]
pos_scores = pos_bd[:, :, :T]                         # [1, T, T] (first T cols)

scores = content_scores + pos_scores                  # [1, T, T]

# Chunk-window mask (left_chunks=1, chunk_size=P=2)
chunk_idx = np.arange(T) // P
diff      = chunk_idx[:, None] - chunk_idx[None, :]
mask      = (diff >= 0) & (diff <= left_chunks)

masked = np.where(mask[None], scores, -np.inf)
mx     = masked.max(axis=2, keepdims=True)
exp_s  = np.exp(masked - mx)
attn   = exp_s / exp_s.sum(axis=2, keepdims=True)

output = np.einsum("bij,bjd->bid", attn, v).astype(np.float32)  # [1, T, 4]

np.savez("io.npz", q=q, k=k, v=v, output=output)
print(f"Saved io.npz  q={q.shape} k={k.shape} v={v.shape} output={output.shape}")

# r_pos as NNEF .dat (128-byte header + f32 data)
data       = r_pos.astype("<f4").tobytes()
dims_bytes = struct.pack("<8I", r_pos.shape[0], r_pos.shape[1], 0, 0, 0, 0, 0, 0)
header = (
    b"\x4e\xef" + struct.pack("<BB", 1, 0)
    + struct.pack("<I", len(data))
    + struct.pack("<I", 2)
    + dims_bytes
    + struct.pack("<I", 32)
    + struct.pack("<HH", 0, 0)
    + b"\x00" * 76
)
assert len(header) == 128
with open("r_pos.dat", "wb") as f:
    f.write(header)
    f.write(data)
print(f"Saved r_pos.dat  shape={r_pos.shape}")
