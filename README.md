# tract benchmark data

Time-series of tract benchmark results, recorded continuously and recovered
from history. This is an **orphan branch** (no shared history with `main`); it
holds data only — no source.

## Layout

    <build-triple>/<device>.json

`index.json` lists the files the nightly bench currently writes.

One file per (build configuration, device). Each file is columnar:

    {
      "start_day": "2023-03-04",
      "commits": [<main commit benched on start_day>, ...],
      "metrics": {
        "<metric>": [<value on start_day>, <start_day + 1 day>, ...]
      }
    }

- Column `i` of every array is the day `start_day + i`; `null` means no run that day.
- `commits` is filled from 2026-06-16 on; `null` where no commit is known.
- Values are rounded to 4 significant figures.
- Metric names: `net.<model>.<reading>.<variant>`, `llm.<model>.<reading>.<backend>`,
  plus `binary_size.*` and `bundle.*`.

## Notes

- `device` is a generic CPU-core or off-the-shelf product label.
- `build-triple` distinguishes builds that genuinely differ in performance
  (e.g. `*-musl` vs `*-gnu`); these are kept as separate series and never merged.
- History begins 2023-03; earlier data predates the current metrics store.
- Points before 2026-06-16 are daily and carry no commit id.
