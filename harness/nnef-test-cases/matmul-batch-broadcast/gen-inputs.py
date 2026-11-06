#!/usr/bin/env python3
"""Generate io.npz for the batch-broadcast matmul synthetic."""

import numpy as np

rng = np.random.default_rng(5)
cases = {
    "lead": ((2, 3, 4, 8), (1, 3, 8, 5), np.float32),
    "trail": ((3, 2, 4, 8), (3, 1, 8, 5), np.float32),
    "mixed": ((1, 8, 1, 1, 2, 64), (1, 8, 1, 2, 64, 6), np.float32),
    "half": ((2, 3, 4, 8), (1, 3, 8, 5), np.float16),
}
io = {}
for name, (sa, sb, dt) in cases.items():
    a = rng.standard_normal(sa).astype(dt)
    b = rng.standard_normal(sb).astype(dt)
    io[f"{name}_a"], io[f"{name}_b"] = a, b
    io[name] = np.matmul(a.astype(np.float32), b.astype(np.float32)).astype(dt)
np.savez("io.npz", **io)
print("Saved io.npz")
