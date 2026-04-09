#!/usr/bin/env python3
"""Generate io.npz and r_pos.dat for ex15."""
import struct, numpy as np

T, Dh, P, left_chunks, B = 8, 4, 4, 3, 1
W = (left_chunks + 1) * P
R = 2 * T - 1

rng = np.random.default_rng(42)
qkv       = rng.standard_normal((B, T, 3*Dh)).astype(np.float32)
r_pos_raw = rng.standard_normal((R, Dh)).astype(np.float32)
W_pos     = np.eye(Dh, dtype=np.float32)  # identity projection
r_pos     = r_pos_raw @ W_pos  # same as r_pos_raw

q, k, v = qkv[:,:,:Dh], qkv[:,:,Dh:2*Dh], qkv[:,:,2*Dh:]
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
mx = masked.max(axis=2, keepdims=True)
exp_s = np.exp(masked - mx)
attn = exp_s / exp_s.sum(axis=2, keepdims=True)
output = np.einsum("bij,bjd->bid", attn, v).astype(np.float32)

np.savez("io.npz", qkv=qkv, output=output)
print(f"Saved io.npz  qkv={qkv.shape} output={output.shape}")

def write_dat(fname, arr):
    data = arr.astype("<f4").tobytes()
    nd = len(arr.shape)
    dims = list(arr.shape) + [0]*(8-nd)
    dims_bytes = struct.pack("<8I", *dims)
    hdr = (b"\x4e\xef" + struct.pack("<BB", 1, 0) + struct.pack("<I", len(data))
        + struct.pack("<I", nd) + dims_bytes + struct.pack("<I", 32) + struct.pack("<HH", 0, 0) + b"\x00" * 76)
    with open(fname, "wb") as f: f.write(hdr); f.write(data)
    print(f"Saved {fname}  shape={arr.shape}")

write_dat("r_pos.dat", r_pos_raw)
write_dat("W_pos.dat", W_pos)
