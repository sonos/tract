#!/usr/bin/env python3

"""
Display the kernel throughputs as a dataframe and a csv file.

Usage: 
1. First, run the benchmarks using `cargo bench -p tract-linalg --features compile_all_kernels --bench kernel_test`.
2. Then run this file in the project root: `python3 linalg/x86_64/kernel_throughput.py`. 

The results are in Gelem/s.
"""

import os
import re
import json
import os.path as path
import pandas as pd

criterion = './target/criterion'
results = os.listdir(criterion)

mat_common_dims = '1024x1000'

df = pd.DataFrame(index=range(16, 256+16, 16), columns=range(1, 33), dtype='float')
for r in results:
    ma = re.match("avx512_mmm_f32_(\d+)x(\d+)", r)
    if not ma:
        continue

    m = int(ma.group(1))
    n = int(ma.group(2))

    path_ = path.join(criterion, r, "f32_cold", f"{mat_common_dims}x{n}")
    benchmark = path.join(path_, "base/benchmark.json")
    with open(benchmark) as f:
        benchmark = json.load(f)

    sample = path.join(path_, "base/sample.json")
    with open(sample) as f:
        sample = json.load(f)

    elements = benchmark["throughput"]["Elements"]
    time_per_iter = sum(sample["times"]) / sum(sample["iters"])

    df.loc[m, n] = round(1 / (time_per_iter / elements), 2)
    print(df.loc[m, n])

pd.set_option('display.max_columns', None)
print(df)
df.to_csv("result.csv")
