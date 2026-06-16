#!/usr/bin/env python3
"""Render the PR-vs-main bench comparison: one comment (movers) + a full job summary.

Fan-in step: consumes every device's result (a dir per device under --results, each
holding `meta.json` = {device, triple} and a `metrics` file) plus the bench-data
checkout (the nightly-main reference), and emits:
  - <out>           : the PR comment markdown (movers only, worst first)
  - $GITHUB_STEP_SUMMARY : the full delta table, grouped by device

Single-shot vs the committed nightly reference; |Δ| must reach the per-class threshold
(.travis/bench-thresholds.json) to count as a mover. Direction-aware: tok/s up is good,
times/sizes/memory up is bad.
"""
import argparse, datetime, glob, json, math, os, re

HIGHER_BETTER = re.compile(r"\.(pp\d+|tg\d+)\.")   # llm throughput; everything else lower-is-better


def read_metrics(path):
    out = {}
    for line in open(path):
        p = line.split()
        if len(p) == 2:
            try:
                # '-' -> '_' to match the recovered history and the nightly references
                # (the old minion ran metric names through `tr '-' '_'` for graphite).
                out[p[0].replace("-", "_")] = float(p[1])
            except ValueError:
                pass
    return out


def threshold_for(metric, cfg):
    for cls, t in cfg["classes"].items():
        if cls in metric:
            return t
    return cfg["default"]


def reference(bench_data, triple, device):
    """latest non-null value per metric from bench-data/<triple>/<device>.json, + its date."""
    path = os.path.join(bench_data, triple, f"{device}.json")
    if not os.path.exists(path):
        return {}, None
    d = json.load(open(path))
    start = datetime.date.fromisoformat(d["start_day"])
    vals, last_idx = {}, -1
    for m, arr in d["metrics"].items():
        for i in range(len(arr) - 1, -1, -1):
            if arr[i] is not None:
                vals[m] = arr[i]
                last_idx = max(last_idx, i)
                break
    ref_day = start + datetime.timedelta(last_idx) if last_idx >= 0 else None
    return vals, ref_day


def humanize(metric):
    p = metric.split(".")
    if p[0] in ("net", "llm") and len(p) >= 4:
        return f"{p[1]} {p[2]} ({p[-1]})"
    if p[0] in ("net", "llm") and len(p) == 3:
        return f"{p[1]} {p[2]}"
    return metric


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="dir of per-device result subdirs")
    ap.add_argument("--bench-data", required=True)
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--pr-sha", required=True)
    ap.add_argument("--out", required=True, help="PR comment markdown path")
    args = ap.parse_args()

    cfg = json.load(open(args.thresholds))
    rows, devices, ref_days = [], [], []
    for meta_path in sorted(glob.glob(os.path.join(args.results, "*", "meta.json"))):
        d = os.path.dirname(meta_path)
        meta = json.load(open(meta_path))
        device, triple = meta["device"], meta["triple"]
        devices.append(device)
        pr = read_metrics(os.path.join(d, "metrics"))
        ref, ref_day = reference(args.bench_data, triple, device)
        if ref_day:
            ref_days.append(ref_day)
        for metric, prv in pr.items():
            rv = ref.get(metric)
            if rv is None or rv == 0:
                continue
            delta = (prv - rv) / rv * 100.0
            higher_better = bool(HIGHER_BETTER.search(metric))
            worse = (delta < 0) if higher_better else (delta > 0)
            thr = threshold_for(metric, cfg)
            rows.append({"device": device, "metric": metric, "ref": rv, "pr": prv,
                         "delta": delta, "worse": worse, "mover": abs(delta) >= thr})

    movers = [r for r in rows if r["mover"]]
    regr = sorted([r for r in movers if r["worse"]], key=lambda r: -abs(r["delta"]))
    impr = sorted([r for r in movers if not r["worse"]], key=lambda r: -abs(r["delta"]))

    ref_day = max(ref_days).isoformat() if ref_days else "n/a"
    age = (datetime.date.today() - max(ref_days)).days if ref_days else "?"
    n_metrics = len(rows)
    devs = ", ".join(f"`{d}`" for d in sorted(set(devices)))

    def fmt(v):
        return f"{v:.4g}"

    def table(items):
        lines = ["| Δ | metric | device | main → PR |", "|---|---|---|---|"]
        for r in items:
            icon = "🔴" if r["worse"] else "🟢"
            lines.append(f"| {icon} **{r['delta']:+.1f}%** | {humanize(r['metric'])} "
                         f"| `{r['device']}` | {fmt(r['ref'])} → {fmt(r['pr'])} |")
        return "\n".join(lines)

    marker = "<!-- bench-vs-main -->"
    if regr:
        head = f"🔴 **Bench vs main — {len(regr)} regression(s)**"
    else:
        head = "✅ **Bench vs main — no regressions**"
    parts = [marker, head, "",
             f"Reference: main nightly, latest **{ref_day}** ({age}d old) · "
             f"PR `{args.pr_sha[:9]}` · ran on {devs} · {n_metrics} metrics compared", ""]
    if regr:
        parts += ["**Regressions** (worst first)", "", table(regr), ""]
    if impr:
        parts += [f"<details><summary>🟢 {len(impr)} improvement(s)</summary>\n\n{table(impr)}\n</details>", ""]
    parts += [f"_advisory · per-class thresholds · single-shot vs nightly reference · "
              f"full table → [run summary]({os.environ.get('RUN_URL', '#')})_"]
    open(args.out, "w").write("\n".join(parts) + "\n")

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as fh:
            fh.write(f"# Bench vs main — full comparison\n\nReference {ref_day} · PR {args.pr_sha[:9]}\n\n")
            for dev in sorted(set(devices)):
                drows = sorted([r for r in rows if r["device"] == dev], key=lambda r: -abs(r["delta"]))
                fh.write(f"## {dev} ({len(drows)} metrics)\n\n| Δ | metric | main → PR |\n|---|---|---|\n")
                for r in drows:
                    flag = "🔴" if (r["worse"] and r["mover"]) else "🟢" if (r["mover"]) else ""
                    fh.write(f"| {flag} {r['delta']:+.1f}% | {humanize(r['metric'])} | {fmt(r['ref'])} → {fmt(r['pr'])} |\n")
                fh.write("\n")
    print(f"regressions={len(regr)} improvements={len(impr)} metrics={n_metrics}")


if __name__ == "__main__":
    main()
