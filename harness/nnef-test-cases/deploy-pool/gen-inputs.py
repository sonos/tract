#!/usr/bin/env python3
"""Generate io.npz: one call of T frames."""

import numpy as np

T = 5
rng = np.random.default_rng(42)

w = np.array(
    [[1.0, 0.5, -0.25], [0.0, 1.0, 0.5], [-0.5, 0.25, 1.0], [0.75, -1.0, 0.25]],
    dtype=np.float32,
)
bias = np.array([[0.5, -0.25, 1.0]], dtype=np.float32)
scale = np.array([[0.5, 2.0, -1.5]], dtype=np.float32)

x = rng.standard_normal((T, 4)).astype(np.float32)
output = np.tanh((x @ w + bias) * scale).sum(axis=1, keepdims=True).astype(np.float32)
np.savez("io.npz", x=x, scale=scale, output=output)
print("Saved io.npz")
