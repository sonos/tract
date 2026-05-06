#!/usr/bin/env python3
"""ex11-subsampled-section — io.npz generator.

T must be chosen so that T is divisible by the audio-frame chunk size
(stride · transformer_chunk = 2 · 2 = 4) and the post-pool dim
1 + (T-3)/2 is divisible by transformer_chunk (= 2).  T=8 works:
post-pool dim = 1 + 5/2 = 1 + 2 = 3 ... actually 3 is odd, doesn't
divide by 2.  T=12: post-pool = 1 + 9/2 = 1+4 = 5, also odd.

The post-subsample dim is `1 + (T-3)/2 = (T-1)/2` (integer div).  For
this to be a multiple of 2 (the transformer chunk), T must be ≡ 1 (mod
4) — but T must also satisfy the audio-frame divisibility.  The
arithmetic doesn't line up, which is itself part of why the substitute
is unclean.  We pick T=12 (post-pool dim = 5) and accept that the
batch reference is the only number we check; pulse fails before
producing output.
"""
import numpy as np

T, D = 12, 4
rng = np.random.default_rng(11)
q = rng.standard_normal((T, D)).astype(np.float32)
k = rng.standard_normal((T, D)).astype(np.float32)
v = rng.standard_normal((T, D)).astype(np.float32)

# Reference batch run reproduces what the graph computes; we only use
# this for the batch leg of the runme.  Pulse leg is expected to fail.
def max_pool_stride2_kernel3(x):
    """[T, D] -> [1+(T-3)/2, D] with kernel=3, stride=2, no padding."""
    out_t = 1 + (T - 3) // 2
    out = np.empty((out_t, x.shape[1]), dtype=x.dtype)
    for i in range(out_t):
        out[i] = x[2*i:2*i+3].max(axis=0)
    return out

q_pre = max_pool_stride2_kernel3(q)
k_pre = max_pool_stride2_kernel3(k)
v_pre = max_pool_stride2_kernel3(v)

scores = q_pre @ k_pre.T
Tc = q_pre.shape[0]
pos = np.arange(Tc)
chunk_id = pos // 2
in_block = (chunk_id[:, None] == chunk_id[None, :])

masked = np.where(in_block, scores * 0.5, -np.inf)
exp_s = np.exp(masked - masked.max(axis=1, keepdims=True))
attn  = exp_s / exp_s.sum(axis=1, keepdims=True)
output = (attn @ v_pre).astype(np.float32)

np.savez("io.npz", q=q, k=k, v=v, output=output)
print(f"Saved io.npz  q={q.shape} k={k.shape} v={v.shape} output={output.shape}  Tc={Tc}")
