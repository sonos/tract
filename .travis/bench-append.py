#!/usr/bin/env python3
"""Append one nightly bench run to the bench-data branch (columnar, one value per line).

Columnar keeps files compact and per-metric (report-friendly). Writing one value per
line (json indent) makes a daily append an INSERTION into each metric array, so git
diffs/commits stay small even though the whole file is rewritten.

File: <out>/<build-triple>/<device>.json
  {"start_day":"YYYY-MM-DD","metrics":{"<metric>":[v0, v1, ...]}}
Column i of every array is start_day + i days; null = no run that day.
"""
import argparse, datetime, json, math, os, sys


def sig(x, n=4):
    if x == 0:
        return 0
    r = round(x, -int(math.floor(math.log10(abs(x)))) + (n - 1))
    return int(r) if r == int(r) else r


def read_metrics(path):
    out = {}
    for line in open(path):
        parts = line.split()
        if len(parts) == 2:
            try:
                out[parts[0]] = sig(float(parts[1]))
            except ValueError:
                pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--out", required=True, help="bench-data checkout root")
    ap.add_argument("--triple", required=True)
    ap.add_argument("--device", required=True)
    ap.add_argument("--day", default=datetime.date.today().isoformat())
    args = ap.parse_args()

    today = read_metrics(args.metrics)
    if not today:
        print(f"no metrics in {args.metrics}", file=sys.stderr)
        sys.exit(1)
    day = datetime.date.fromisoformat(args.day)

    path = os.path.join(args.out, args.triple, f"{args.device}.json")
    if os.path.exists(path):
        data = json.load(open(path))
        start = datetime.date.fromisoformat(data["start_day"])
        arrays = data["metrics"]
    else:
        start, arrays = day, {}

    idx = (day - start).days
    if idx < 0:
        print(f"day {day} precedes start_day {start}", file=sys.stderr)
        sys.exit(1)

    n = idx + 1
    for k in set(arrays) | set(today):
        a = arrays.get(k, [])
        if len(a) < n:                       # null-pad skipped days / new metric's prefix
            a = a + [None] * (n - len(a))
        a[idx] = today.get(k)                # value today, else null (metric not run today)
        arrays[k] = a

    obj = {"start_day": start.isoformat(),
           "metrics": {k: arrays[k] for k in sorted(arrays)}}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=0)         # one value per line -> insertion-friendly diffs
        fh.write("\n")
    print(f"appended {args.device} {args.day}: {len(today)} metrics, span now {n} days")


if __name__ == "__main__":
    main()
