#!/usr/bin/env python3
import struct, numpy as np

T, Dh, P, left_chunks, B = 16, 4, 4, 3, 1
W = (left_chunks + 1) * P  # 16
R = 2 * T - 1  # 31
center = T  # 16

rng = np.random.default_rng(42)
q     = rng.standard_normal((B, T, Dh)).astype(np.float32)
k     = rng.standard_normal((B, T, Dh)).astype(np.float32)
v     = rng.standard_normal((B, T, Dh)).astype(np.float32)
r_full = rng.standard_normal((R, Dh)).astype(np.float32)

r_pos = r_full[center - T : center + T - 1]  # [31, 4] = all of r_full
content = np.einsum("bid,bjd->bij", q, k)
pos_raw = np.einsum("bid,jd->bij", q, r_pos)
pp = np.pad(pos_raw, [[0,0],[0,0],[1,0]])
pv = pp.reshape(B, -1, T)
ps = pv[:, 1:2*T, :]
pb = ps.reshape(B, T, -1)
pos_scores = pb[:, :, :T]
scores = content + pos_scores

ci = np.arange(T) // P
diff = ci[:, None] - ci[None, :]
mask = (diff >= 0) & (diff <= left_chunks)
masked = np.where(mask[None], scores, -np.inf)
mx = masked.max(axis=2, keepdims=True)
exp_s = np.exp(masked - mx)
attn = exp_s / exp_s.sum(axis=2, keepdims=True)
output = np.einsum("bij,bjd->bid", attn, v).astype(np.float32)

np.savez("io.npz", q=q, k=k, v=v, output=output)
print(f"io.npz  q={q.shape} k={k.shape} v={v.shape} output={output.shape}")

data = r_full.astype("<f4").tobytes()
dims = struct.pack("<8I", R, Dh, 0, 0, 0, 0, 0, 0)
hdr = (b"\x4e\xef" + struct.pack("<BB", 1, 0) + struct.pack("<I", len(data))
    + struct.pack("<I", 2) + dims + struct.pack("<I", 32) + struct.pack("<HH", 0, 0) + b"\x00" * 76)
with open("r_pos.dat", "wb") as f: f.write(hdr); f.write(data)
print(f"r_pos.dat  shape={r_full.shape}")
