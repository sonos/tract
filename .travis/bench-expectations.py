#!/usr/bin/env python3
"""Emit per-metric expected values for one (triple, device), for the bundle's
expectation-guided retry.

One `metric value` line per metric; value = recent median of the non-null points in
bench-data/<triple>/<device>.json. Keys are underscored, as stored in bench-data; the
bundle normalizes '-' -> '_' before looking a metric up. A device with no history
yields an empty file, which simply disables retry there (single-shot, as before).
"""
import argparse, json, os, statistics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-data", required=True)
    ap.add_argument("--triple", required=True)
    ap.add_argument("--device", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=10, help="trailing nights to median over")
    a = ap.parse_args()

    path = os.path.join(a.bench_data, a.triple, f"{a.device}.json")
    lines = []
    if os.path.exists(path):
        d = json.load(open(path))
        for m, arr in d.get("metrics", {}).items():
            vals = [v for v in arr[-a.window:] if v is not None]
            if vals:
                lines.append(f"{m} {statistics.median(vals)}")
    open(a.out, "w").write("".join(f"{l}\n" for l in lines))
    print(f"expectations: {len(lines)} metrics -> {a.out}")


if __name__ == "__main__":
    main()
