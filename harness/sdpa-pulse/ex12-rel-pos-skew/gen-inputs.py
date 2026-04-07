#!/usr/bin/env python3
"""Generate io.npz and r_full.dat for ex12-rel-pos-skew.

Transformer-XL relative-position attention with the skew trick.

Parameters: T=8, P=2, Dh=4, H=1, left_chunks=1, max_rel=15 (= 2*T_max - 1)

The position-score path:
    Q[T, Dh] @ R^T[2T-1, Dh] -> [T, 2T-1]
    -> pad left by 1 -> [T, 2T]
    -> reshape [T, 2T] -> [2T, T]
    -> slice rows 1..2T-1 -> [2T-1, T]
    -> reshape [2T-1, T] -> [T, 2T-1]
    -> slice cols :T -> [T, T]   (the skew trick)

Combined with content score Q @ K^T, masked with a chunk-window mask,
softmax, and weighted sum with V.

Input:  qkv   [1, T, 3*Dh] = [1, 8, 12]  (batch dim 1, streaming)
Output: [1, T, Dh] = [1, 8, 4]

r_full [max_rel, Dh] = [15, 4] is a model variable saved to r_full.dat.
"""

import struct
import numpy as np

T, Dh, P, left_chunks, B = 8, 4, 2, 0, 1
MAX_T = T                     # single sequence length in this test
max_rel = 2 * MAX_T - 1       # 15

rng = np.random.default_rng(42)

q = rng.standard_normal((B, T, Dh)).astype(np.float32)
k = rng.standard_normal((B, T, Dh)).astype(np.float32)
v = rng.standard_normal((B, T, Dh)).astype(np.float32)

# Positional encoding table [max_rel, Dh] — stored as model variable r_full.dat
r_full = rng.standard_normal((max_rel, Dh)).astype(np.float32)

# Content scores: Q @ K^T  [B, T, T]
content_scores = np.einsum("bid,bjd->bij", q, k)

# Dynamic slice of R for the current T: R = r_full[center-T : center+T-1]
# center = max_rel // 2 + 1 = 8  (same convention as the encoder)
# end is exclusive; len = (S+7) - (8-S) = 2S-1
center = max_rel // 2 + 1      # = 8
begin  = center - T            # = 0
end    = center + T - 1        # = 15 (exclusive)
r = r_full[begin:end]          # [2T-1, Dh] = [15, 4]

# Position scores: Q @ R^T  [B, T, 2T-1]
pos_raw = np.einsum("bid,jd->bij", q, r)   # [B, T, 2T-1]

# Skew trick (reshape variant — matches graph.nnef and the encoder)
pos_padded  = np.pad(pos_raw, ((0,0),(0,0),(1,0)))  # [B, T, 2T]
pos_view    = pos_padded.reshape(B, 2*T, T)          # [B, 2T, T]
pos_sliced  = pos_view[:, 1:, :]                     # [B, 2T-1, T]
pos_bd      = pos_sliced.reshape(B, T, 2*T-1)        # [B, T, 2T-1]
pos_scores  = pos_bd[:, :, :T]                       # [B, T, T]

scores = content_scores + pos_scores  # [B, T, T]

# Chunk-window mask (same as ex04/ex09)
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

# ── Save r_full as NNEF .dat variable (model weight) ──────────────────────
# NNEF tensor binary format: 128-byte header + little-endian f32 data.
# Header layout (all LE):
#   [0:2]   magic      = [0x4e, 0xef]
#   [2]     version_maj = 1
#   [3]     version_min = 0
#   [4:8]   data_size_bytes (u32)
#   [8:12]  rank (u32)
#   [12:44] dims[8] (u32 each)
#   [44:48] bits_per_item (u32)
#   [48:50] item_type (u16): 0 = IEEE float
#   [50:52] item_type_vendor (u16): 0 = standard
#   [52:84] item_type_params_deprecated [32 bytes, zero]
#   [84:128] padding [11 u32, zero]
data   = r_full.astype("<f4").tobytes()
# Reassemble header correctly (128 bytes)
dims_bytes = struct.pack("<8I", r_full.shape[0], r_full.shape[1], 0, 0, 0, 0, 0, 0)
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

with open("r_full.dat", "wb") as f:
    f.write(header)
    f.write(data)
print(f"Saved r_full.dat  shape={r_full.shape}")
