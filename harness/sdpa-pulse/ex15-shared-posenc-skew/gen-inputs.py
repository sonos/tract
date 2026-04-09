#!/usr/bin/env python3
"""Generate io.npz, posEnc.dat, W_pos.dat, W_pos2.dat for ex15."""

import struct
import numpy as np

T, H, Dh, P, left_chunks, B = 8, 2, 4, 2, 1, 1
D_model = H * Dh  # 8
W = (left_chunks + 1) * P  # 4
R = 2 * T - 1  # 15
center = T  # 8

rng = np.random.default_rng(42)

q      = rng.standard_normal((B, T, D_model)).astype(np.float32)
k      = rng.standard_normal((B, T, D_model)).astype(np.float32)
v      = rng.standard_normal((B, T, D_model)).astype(np.float32)
posEnc_full = rng.standard_normal((R, D_model)).astype(np.float32)
W_pos  = rng.standard_normal((D_model, D_model)).astype(np.float32)
W_pos2 = rng.standard_normal((D_model, D_model)).astype(np.float32)

# Dynamic slice: posEnc = posEnc_full[center-T : center+T-1]
posEnc = posEnc_full[center - T : center + T - 1]  # [2*T-1, D_model] = [15, 8]

q_mh = q.reshape(B, T, H, Dh)
k_mh = k.reshape(B, T, H, Dh)
v_mh = v.reshape(B, T, H, Dh)

content_scores = np.einsum("bihd,bjhd->bhij", q_mh, k_mh)

linearPos = posEnc @ W_pos
r_pos_view = linearPos.reshape(-1, H, Dh)
pos_raw = np.einsum("bihd,jhd->bhij", q_mh, r_pos_view)

# Skew
pos_padded = np.pad(pos_raw, [[0,0],[0,0],[0,0],[1,0]])
pos_view_r = pos_padded.reshape(B, H, -1, T)
pos_sliced = pos_view_r[:, :, 1:2*T, :]
pos_bd = pos_sliced.reshape(B, H, T, -1)
pos_scores = pos_bd[:, :, :, :T]

# Dummy layer 2 (zeroed out)
scores = content_scores + pos_scores

# Chunk-window mask
chunk_idx = np.arange(T) // P
diff = chunk_idx[:, None] - chunk_idx[None, :]
mask = (diff >= 0) & (diff <= left_chunks)

masked = np.where(mask[None, None], scores, -np.inf)
mx = masked.max(axis=3, keepdims=True)
exp_s = np.exp(masked - mx)
attn = exp_s / exp_s.sum(axis=3, keepdims=True)

ctx = np.einsum("bhij,bjhd->bihd", attn, v_mh)
output = ctx.reshape(B, T, D_model).astype(np.float32)

np.savez("io.npz", q=q, k=k, v=v, output=output)
print(f"Saved io.npz  q={q.shape} k={k.shape} v={v.shape} output={output.shape}")

def write_dat(filename, array):
    data = array.astype("<f4").tobytes()
    ndims = len(array.shape)
    dims = list(array.shape) + [0] * (8 - ndims)
    dims_bytes = struct.pack("<8I", *dims)
    header = (
        b"\x4e\xef" + struct.pack("<BB", 1, 0)
        + struct.pack("<I", len(data))
        + struct.pack("<I", ndims)
        + dims_bytes
        + struct.pack("<I", 32)
        + struct.pack("<HH", 0, 0)
        + b"\x00" * 76
    )
    assert len(header) == 128
    with open(filename, "wb") as f:
        f.write(header)
        f.write(data)
    print(f"Saved {filename}  shape={array.shape}")

write_dat("posEnc.dat", posEnc_full)
write_dat("W_pos.dat", W_pos)
write_dat("W_pos2.dat", W_pos2)
