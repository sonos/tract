#!/usr/bin/env python3
"""ex11-blockified — io.npz generator.

Audio shape is [4*s, D] with s=2 → T=8.  This is the *streaming-first*
form: input is exactly 4 audio frames per chunk, no boundary leak.
The pool's internal `(0,1)` after-pad supplies the kernel=3 boundary
frame as a zero, both in batch and in pulse.

Reference computation mirrors what the model does:
  pad_after audio with one zero, max_pool kernel=3 stride=2 → Tc=2*s,
  reshape [Tc, D] → [s, P=2, D], per-chunk Q·Kᵀ scaled softmax · V,
  reshape back to [Tc, D].
"""
import numpy as np

s = 2
T = 4 * s   # 8
D = 4
P = 2       # transformer chunk size
rng = np.random.default_rng(11)
q = rng.standard_normal((T, D)).astype(np.float32)
k = rng.standard_normal((T, D)).astype(np.float32)
v = rng.standard_normal((T, D)).astype(np.float32)


def max_pool_kernel3_stride2_padafter1(x):
    """[T, D] -> [Tc, D] with kernel=3, stride=2, pad-after 1 (valid mode).

    Tract's max_pool only visits valid (non-padded) positions, so the
    after-pad widens the geometry but doesn't contribute to the max.
    The last window thus reduces over 2 real frames instead of 3.
    """
    T_in = x.shape[0]
    Tp = T_in + 1                      # padded geometry
    out_t = 1 + (Tp - 3) // 2          # 2s
    out = np.empty((out_t, x.shape[1]), dtype=x.dtype)
    for i in range(out_t):
        start = 2 * i
        end = min(start + 3, T_in)     # clamp to real frames
        out[i] = x[start:end].max(axis=0)
    return out


q_pre = max_pool_kernel3_stride2_padafter1(q)
k_pre = max_pool_kernel3_stride2_padafter1(k)
v_pre = max_pool_kernel3_stride2_padafter1(v)
Tc = q_pre.shape[0]                    # 2*s = 4

# Per-chunk SDPA on [s, P, D].
q_ch = q_pre.reshape(s, P, D)
k_ch = k_pre.reshape(s, P, D)
v_ch = v_pre.reshape(s, P, D)

scores = np.einsum("cid,cjd->cij", q_ch, k_ch)
scaled = scores * 0.5
exp_s = np.exp(scaled - scaled.max(axis=-1, keepdims=True))
attn = exp_s / exp_s.sum(axis=-1, keepdims=True)
out_ch = np.einsum("cij,cjd->cid", attn, v_ch)
output = out_ch.reshape(2 * s, D).astype(np.float32)

np.savez("io.npz", q=q, k=k, v=v, output=output)
print(f"Saved io.npz  q={q.shape} k={k.shape} v={v.shape} output={output.shape}  Tc={Tc}")
