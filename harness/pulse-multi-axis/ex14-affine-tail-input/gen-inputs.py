#!/usr/bin/env python3
"""ex14-affine-tail-input — io.npz generator.

T = 4, P = 2, D = 4.  Inputs are padded by 1 leading zero so the
streaming axis seen by the section is `1+T = 5`, with chunks
`[0, 0, 0, 1, 1]` — chunk 0 has 3 elements (1 padding + 2 data),
chunk 1 has 2 elements.
"""
import numpy as np

T, D, P = 4, 4, 2
T_full = 1 + T
rng = np.random.default_rng(14)

# `a` and `b` sized to the post-affine streaming dim (1+T = 5) so the
# external matches the graph's `[1+T, 4]` declaration directly.
a = rng.standard_normal((T_full, D)).astype(np.float32)
b = rng.standard_normal((T_full, D)).astype(np.float32)

scores = np.einsum("id,jd->ij", a, b)                                    # [1+T, 1+T]

idx       = np.arange(T_full)
chunk_id  = idx // P
mask      = (chunk_id[:, None] == chunk_id[None, :]).astype(np.float32)  # [1+T, 1+T]

masked    = scores * mask
output    = masked.sum(axis=0).astype(np.float32)                        # [1+T]

np.savez("io.npz", a=a, b=b, output=output)
print(f"Saved io.npz  a={a.shape} b={b.shape} output={output.shape}")
