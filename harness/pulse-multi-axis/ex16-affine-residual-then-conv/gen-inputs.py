#!/usr/bin/env python3
"""ex16-affine-residual-then-conv — io.npz generator.

T=4 input.  Stride-2 max_pool with (1, 0) leading padding + kernel 3
gives output `1+(T-1)/2 = 1+(3)/2 = 1+1 = 2` for T=4? Actually
output = (T + 1 - 3)/2 + 1 = (T-2)/2 + 1.  For T=4: (2)/2+1 = 2.

Reference computation in numpy mirrors the graph.
"""
import numpy as np

T = 4
rng = np.random.default_rng(16)
x = rng.standard_normal((T,)).astype(np.float32)

# Reference: pad x with 1 leading zero, add arange(0, 1+T), then
# stride-1 max_pool with symmetric (1, 1) padding.
x_pad    = np.concatenate([np.zeros(1, dtype=np.float32), x])     # [1+T]
rng_f32  = np.arange(1 + T, dtype=np.float32)
combined = x_pad + rng_f32                                        # [1+T]
combined_pad = np.concatenate([np.zeros(1, dtype=np.float32), combined, np.zeros(1, dtype=np.float32)])
y = np.array([combined_pad[i:i+3].max() for i in range(1 + T)], dtype=np.float32)

np.savez("io.npz", x=x, y=y)
print(f"Saved io.npz  x={x.shape} y={y.shape}")
