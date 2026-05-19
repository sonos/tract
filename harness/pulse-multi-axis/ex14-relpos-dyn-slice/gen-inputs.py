#!/usr/bin/env python3
"""ex14-relpos-dyn-slice — io.npz generator.

T = 4, chunk size P = 2 → 2 chunks of 2 tokens each.
Banded-causal mask with L = 1 → each row at chunk c attends to chunks {c-1, c}.

Reference computation (matches graph.nnef):
    pos_slice         = pos_enc_pe[5 - T : 4 + T]                  # [2T-1, D]
    rel_pos[i, j]     = sum_d q[i, d] * pos_slice[j, d]            # [T, 2T-1]
    skewed[i, j]      = "skew trick"(rel_pos)[i, j]
                      = rel_pos[i, (T-1) + j - i]  if in bounds else 0   # [T, T]
    content[i, j]     = sum_d q[i, d] * k[j, d]                    # [T, T]
    scores            = content + skewed                           # [T, T]
    in_band[i, j]     = (-1 <= i//P - j//P <= 0)                   # banded-causal
    masked            = where(in_band, scores, -inf)
    attn              = softmax(masked, axis=-1)                   # [T, T]
    output            = attn @ v                                   # [T, D]
"""
import numpy as np
import struct

T, D, P, T_MAX = 4, 4, 2, 5
R_MAX = 2 * T_MAX - 1   # 9
CENTRE = T_MAX - 1      # 4


def write_nnef_dat(path, tensor):
    """Write a tract-compatible NNEF .dat file (header layout from
    tract/nnef/src/tensors.rs::Header)."""
    assert tensor.dtype == np.float32, tensor.dtype
    shape = tensor.shape
    rank = len(shape)
    assert rank <= 8, rank
    data = tensor.tobytes()
    dims = list(shape) + [0] * (8 - rank)
    header = (
        bytes([0x4E, 0xEF])               # magic
        + bytes([1, 0])                    # version_maj, version_min
        + struct.pack("<I", len(data))     # data_size_bytes
        + struct.pack("<I", rank)
        + struct.pack("<8I", *dims)
        + struct.pack("<I", 32)            # bits_per_item (f32)
        + struct.pack("<H", 0)             # item_type (0 = IEEE float)
        + struct.pack("<H", 0)             # item_type_vendor
        + bytes(32)                        # item_type_params_deprecated
        + bytes(44)                        # padding (11 × u32)
    )
    assert len(header) == 128, len(header)
    with open(path, "wb") as f:
        f.write(header)
        f.write(data)

rng = np.random.default_rng(14)

q          = rng.standard_normal((T, D)).astype(np.float32)
k          = rng.standard_normal((T, D)).astype(np.float32)
v          = rng.standard_normal((T, D)).astype(np.float32)
pos_enc_pe = rng.standard_normal((R_MAX, D)).astype(np.float32)

# Slice pos_enc_pe at [5 - T, 4 + T) → 2T-1 rows centred on row 4.
begin, end = 5 - T, 4 + T
pos_slice = pos_enc_pe[begin:end]                         # [2T-1, D]
assert pos_slice.shape == (2 * T - 1, D), pos_slice.shape

# Rel-pos contribution (einsum "id,jd->ij").
rel_pos = q @ pos_slice.T                                 # [T, 2T-1]

# Skew trick: out[i, j] = rel_pos[i, (T-1) + j - i].
skewed = np.zeros((T, T), dtype=np.float32)
for i in range(T):
    for j in range(T):
        col = (T - 1) + j - i
        if 0 <= col < 2 * T - 1:
            skewed[i, j] = rel_pos[i, col]

content = q @ k.T                                         # [T, T]
scores  = content + skewed                                # [T, T]

# Banded-causal (lookback) mask, P = 2, L = 1.
idx       = np.arange(T)
chunk_id  = idx // P
diff      = chunk_id[:, None] - chunk_id[None, :]
in_band   = (diff >= 0) & (diff <= 1)                     # [T, T]

masked = np.where(in_band, scores, -np.inf)

# Softmax row-by-row (with guard for fully-masked rows).
m     = masked.max(axis=-1, keepdims=True)
exp_s = np.exp(masked - m)
denom = exp_s.sum(axis=-1, keepdims=True)
attn  = np.where(denom > 0, exp_s / denom, 0.0).astype(np.float32)

output = (attn @ v).astype(np.float32)

np.savez("io.npz", q=q, k=k, v=v, output=output)
write_nnef_dat("pos_enc_pe.dat", pos_enc_pe)
print(
    f"Saved io.npz  q={q.shape} output={output.shape}; "
    f"pos_enc_pe.dat shape={pos_enc_pe.shape}"
)
