#!/usr/bin/env python3
"""Generate io.npz for the rank-5 cross-broadcast binary synthetic."""

import numpy as np

rng = np.random.default_rng(7)
a = rng.standard_normal((1, 1, 1, 4, 1)).astype(np.float32)
b = rng.standard_normal((1, 1, 1, 1, 60)).astype(np.float32)

np.savez("io.npz", a=a, b=b, output=(a - b).astype(np.float32))
print("Saved io.npz")
