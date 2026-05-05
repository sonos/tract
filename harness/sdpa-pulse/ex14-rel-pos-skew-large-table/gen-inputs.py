#!/usr/bin/env python3
"""Generate io.npz and r_pos.dat for ex14-rel-pos-skew-large-table.

Transformer-XL relative-position attention with the skew trick.
The position table r_pos is a FIXED VARIABLE of shape [2*T-1, Dh] = [15, 4].
This models the Nemotron encoder where the RPE table is pre-computed for the
full sequence length T and loaded as a constant (not dynamically sliced from a
larger table as in ex12/ex13).

Parameters: T=8, P=2, Dh=4, H=1, left_chunks=1, W=(left_chunks+1)*P=4

At pulse time: r_pos stays [15, 4] (not [2*P-1=3, 4] as in ex13/ex12).
This causes the skew to produce wrong-sized pos_scores in the pulsed model.

Input:  qkv   [1, T, 3*Dh] = [1, 8, 12]  (batch dim 1, streaming)
Output: [1, T, Dh] = [1, 8, 4]

r_pos [2*T-1, Dh] = [15, 4] is a model variable saved to r_pos.dat.
"""

import struct
import numpy as np

T, Dh, P, left_chunks, B = 8, 4, 2, 1, 1
W = (left_chunks + 1) * P   # 4
max_rel = 2 * T - 1          # 15

rng = np.random.default_rng(42)

q = rng.standard_normal((B, T, Dh)).astype(np.float32)
k = rng.standard_normal((B, T, Dh)).astype(np.float32)
v = rng.standard_normal((B, T, Dh)).astype(np.float32)

# Fixed position encoding table [2*T-1, Dh]
# Centered so r_pos[T-1] is the zero-relative-position entry.
r_pos = rng.standard_normal((max_rel, Dh)).astype(np.float32)

# Content scores: Q @ K^T  [B, T, T]
content_scores = np.einsum("bid,bjd->bij", q, k)

# Position scores: Q @ R^T  [B, T, 2T-1]
pos_raw = np.einsum("bid,jd->bij", q, r_pos)   # [B, T, 15]

# Skew trick (T=8, T_shape=T at batch time)
pos_padded  = np.pad(pos_raw, ((0,0),(0,0),(1,0)))  # [B, T, 2T]
pos_view    = pos_padded.reshape(B, 2*T, T)          # [B, 2T, T]
pos_sliced  = pos_view[:, 1:, :]                     # [B, 2T-1, T]
pos_bd      = pos_sliced.reshape(B, T, 2*T-1)        # [B, T, 2T-1]
pos_scores  = pos_bd[:, :, :T]                       # [B, T, T]

scores = content_scores + pos_scores  # [B, T, T]

# Chunk-window mask with left_chunks=1
chunk_idx = np.arange(T) // P
diff      = chunk_idx[:, None] - chunk_idx[None, :]   # [T, T]
mask      = (diff >= 0) & (diff <= left_chunks)        # [T, T]

masked = np.where(mask[None], scores, -np.inf)         # [B, T, T]

# Stable softmax over axis 2
mx    = masked.max(axis=2, keepdims=True)
exp_s = np.exp(masked - mx)
attn  = exp_s / exp_s.sum(axis=2, keepdims=True)      # [B, T, T]

# Weighted sum: [B, T, Dh]
output = np.einsum("bij,bjd->bid", attn, v).astype(np.float32)

# ── Save model input/output ────────────────────────────────────────────────
qkv = np.concatenate([q, k, v], axis=2)               # [B, T, 3*Dh]
np.savez("io.npz", qkv=qkv, output=output)
print(f"Saved io.npz  qkv={qkv.shape} output={output.shape}")

# ── Save r_pos as NNEF .dat variable (model weight) ───────────────────────
# NNEF tensor binary format: 128-byte header + little-endian f32 data.
data   = r_pos.astype("<f4").tobytes()
dims_bytes = struct.pack("<8I", r_pos.shape[0], r_pos.shape[1], 0, 0, 0, 0, 0, 0)
header = (
    b"\x4e\xef"                                  # magic [0:2]
    + struct.pack("<BB", 1, 0)                   # version [2:4]
    + struct.pack("<I", len(data))               # data_size_bytes [4:8]
    + struct.pack("<I", 2)                       # rank [8:12]
    + dims_bytes                                 # dims [12:44]
    + struct.pack("<I", 32)                      # bits_per_item [44:48]
    + struct.pack("<HH", 0, 0)                   # item_type, vendor [48:52]
    + b"\x00" * 32                              # params_deprecated [52:84]
    + b"\x00" * 44                              # padding [84:128]
)
assert len(header) == 128, f"Header length {len(header)} != 128"

with open("r_pos.dat", "wb") as f:
    f.write(header)
    f.write(data)
print(f"Saved r_pos.dat  shape={r_pos.shape}")
