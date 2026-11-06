#!/usr/bin/env python3
"""Generate io.npz for the softmax over -inf masked rows."""

import numpy as np

rng = np.random.default_rng(3)
x = rng.standard_normal((5, 64)).astype(np.float32)
x[0, 1:] = -np.inf
x[1, :-1] = -np.inf
x[2, :60] = -np.inf
x[3, ::3] = -np.inf
e = np.exp(x - x.max(axis=1, keepdims=True))
np.savez("io.npz", x=x, y=(e / e.sum(axis=1, keepdims=True)).astype(np.float32))
print("Saved io.npz")
