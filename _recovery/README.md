# bench-data recovery scripts

One-off tooling that produced this branch's historical series by recovering
tract's benchmark history from a time-series metrics store (2023-03 onward,
before retention rolled it off). Archival/reference — the ongoing benchmark
pipeline does not use these.

## Run order
- `discover.py`  — enumerate the metric tree, machines, and main/master spans.
- `capture.py`   — raw full-resolution pull of every real-data series (resumable).
- `translate.py` — null-strip, map machine ids -> generic device names, drop
  non-public metrics, emit the columnar `<build-triple>/<device>.json` tree.

## Config
All site-specific/private values (endpoint, machine->device map, excludes) live
in `recovery-config.json`, which is **not committed** (see `.gitignore`). Copy
`recovery-config.example.json`, fill it, or point `$BENCH_RECOVERY_CONFIG` at it.
