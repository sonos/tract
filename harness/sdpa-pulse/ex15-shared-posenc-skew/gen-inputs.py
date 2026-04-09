#!/usr/bin/env python3
import struct, numpy as np

T, Dh, P, left_chunks, B = 8, 4, 4, 3, 1
W = (left_chunks + 1) * P
R = 2 * T - 1

rng = np.random.default_rng(42)
q     = rng.standard_normal((B, T, Dh)).astype(np.float32)
k     = rng.standard_normal((B, T, Dh)).astype(np.float32)
r_pos = rng.standard_normal((R, Dh)).astype(np.float32)

content = np.einsum("bid,bjd->bij", q, k)
pos_raw = np.einsum("bid,jd->bij", q, r_pos)
pos_padded = np.pad(pos_raw, [[0,0],[0,0],[1,0]])
pos_view   = pos_padded.reshape(B, -1, T)
pos_sliced = pos_view[:, 1:2*T, :]
pos_bd     = pos_sliced.reshape(B, T, -1)
pos_scores = pos_bd[:, :, :T]
scores = content + pos_scores

chunk_idx = np.arange(T) // P
diff = chunk_idx[:, None] - chunk_idx[None, :]
mask = (diff >= 0) & (diff <= left_chunks)
masked = np.where(mask[None], scores, -np.inf)
output = np.exp(masked - masked.max(axis=2, keepdims=True))
output = (output / output.sum(axis=2, keepdims=True)).astype(np.float32)

np.savez("io.npz", q=q, k=k, output=output)
print(f"io.npz  q={q.shape} k={k.shape} output={output.shape}")

data = r_pos.astype("<f4").tobytes()
dims = struct.pack("<8I", R, Dh, 0, 0, 0, 0, 0, 0)
hdr = (b"\x4e\xef" + struct.pack("<BB", 1, 0) + struct.pack("<I", len(data))
    + struct.pack("<I", 2) + dims + struct.pack("<I", 32) + struct.pack("<HH", 0, 0) + b"\x00" * 76)
with open("r_pos.dat", "wb") as f: f.write(hdr); f.write(data)
print(f"r_pos.dat  shape={r_pos.shape}")
