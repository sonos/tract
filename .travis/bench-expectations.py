#!/usr/bin/env python3
"""Emit per-metric expectations for one (triple, device), for the bundle's
expectation-guided retry.

One `metric expected threshold` line per *gated* metric, from
bench-data/<triple>/<device>.json:
  - expected  = recent median of the non-null points
  - threshold = the |Δ%| that would make this metric a PR red (bench_common.red_threshold)

So the bundle re-runs exactly the benches whose measured value would show red — every
red the maintainer sees has been confirmed across re-runs. Metrics that are never gated
(operational, sub-resolution, or noisy-without-history) are omitted, so they aren't
retried. Keys are underscored, as in bench-data; the bundle normalizes '-' -> '_'.
A device with no history yields an empty file (retry disabled there, single-shot).
"""
import argparse, json, os, statistics
import bench_common as bc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-data", required=True)
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--triple", required=True)
    ap.add_argument("--device", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=10, help="trailing nights to median over")
    a = ap.parse_args()

    cfg = bc.load_cfg(a.thresholds)
    path = os.path.join(a.bench_data, a.triple, f"{a.device}.json")
    lines = []
    if os.path.exists(path):
        d = json.load(open(path))
        for m, arr in d.get("metrics", {}).items():
            vals = [v for v in arr[-a.window:] if v is not None]
            if not vals:
                continue
            expected = statistics.median(vals)
            thr = bc.red_threshold(m, cfg, bc.series_noise(arr), expected)
            if thr is not None:
                lines.append(f"{m} {expected} {thr}")
    open(a.out, "w").write("".join(f"{l}\n" for l in lines))
    print(f"expectations: {len(lines)} gated metrics -> {a.out}")


if __name__ == "__main__":
    main()
