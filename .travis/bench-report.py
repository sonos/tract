#!/usr/bin/env python3
"""Render the PR-vs-main bench comparison: one comment (movers) + a full job summary.

Fan-in step: consumes every device's result (a dir per device under --results, each
holding `meta.json` = {device, triple} and a `metrics` file) plus the bench-data
checkout (the nightly-main reference), and emits:
  - <out>           : the PR comment markdown (movers only, worst first)
  - $GITHUB_STEP_SUMMARY : the full delta table, grouped by device

Single-shot vs the committed nightly reference; |Δ| must reach an adaptive threshold
(.travis/bench-thresholds.toml) to count as a mover. The threshold is
max(class_floor, k * noise), where `noise` is the series' own recent day-to-day
dispersion in the reference — so a noisy (host, model) self-widens and a clean one
stays tight, with no metric named or excluded by hand. Direction-aware: tok/s up is
good, times/sizes/memory up is bad.
"""
import argparse, datetime, glob, json, os, re, tomllib

HIGHER_BETTER = re.compile(r"\.(pp\d+|tg\d+)\.")   # llm throughput; everything else lower-is-better
SPEED = re.compile(r"\.(evaltime|pp\d+|tg\d+)\.")  # the merge signal: inference speed, shown first


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


def floor_for(metric, floors):
    for cls, t in floors.items():
        if cls != "default" and cls in metric:
            return t
    return floors.get("default", 5)


def threshold_for(metric, cfg, noise_pct):
    """Adaptive: max(class floor, k * the series' own historical noise). Falls back to
    the floor alone when there's too little history to estimate noise."""
    floor = floor_for(metric, cfg["floors"])
    if noise_pct is None:
        return floor
    return max(floor, cfg.get("k", 3.0) * noise_pct)


def series_noise(arr, window=40, min_pairs=8):
    """p90 of recent day-to-day |Δ%| for a metric series — its intrinsic run-to-run
    dispersion. p90 (not max) so a stray real-change spike doesn't inflate it; None
    when there isn't enough history to judge."""
    d, prev = [], None
    for v in arr[-window:]:
        if v is None:
            prev = None
            continue
        if prev not in (None, 0):
            d.append(abs(v - prev) / abs(prev) * 100.0)
        prev = v
    if len(d) < min_pairs:
        return None
    d.sort()
    return d[min(len(d) - 1, int(0.9 * len(d)))]


def reference(bench_data, triple, device):
    """latest non-null value + recent noise (p90 day-to-day |Δ%|) per metric from
    bench-data/<triple>/<device>.json, plus the reference date."""
    path = os.path.join(bench_data, triple, f"{device}.json")
    if not os.path.exists(path):
        return {}, {}, None
    d = json.load(open(path))
    start = datetime.date.fromisoformat(d["start_day"])
    vals, noise, last_idx = {}, {}, -1
    for m, arr in d["metrics"].items():
        noise[m] = series_noise(arr)
        for i in range(len(arr) - 1, -1, -1):
            if arr[i] is not None:
                vals[m] = arr[i]
                last_idx = max(last_idx, i)
                break
    ref_day = start + datetime.timedelta(last_idx) if last_idx >= 0 else None
    return vals, noise, ref_day


# metric type-token -> (human label, unit); first substring match wins.
TYPE_INFO = [
    ("evaltime",                 ("evaltime",      "s")),
    ("time_to_model_ready",      ("load+optimize", "s")),
    ("time_to_before_optimize",  ("load",          "s")),
    ("rsz_at_model_ready",       ("RSS @ ready",   "mem")),
    ("rsz_at_before_optimize",   ("RSS @ load",    "mem")),
    ("active_at_model_ready",    ("heap @ ready",  "mem")),
    ("active_at_before_optimize",("heap @ load",   "mem")),
    ("pp512",                    ("prefill",       "tok")),
    ("tg128",                    ("decode",        "tok")),
    ("bench_runtime",            ("bench wall",    "s")),
    ("binary_size",              ("binary size",   "mem")),
]


def describe(metric):
    """(model, label, variant, unit) for a metric key kind.model.type.variant."""
    p = metric.split(".")
    label, unit = None, "raw"
    for key, (lbl, u) in TYPE_INFO:
        if key in metric:
            label, unit = lbl, u
            break
    if p[0] in ("net", "llm"):
        model = p[1] if len(p) > 1 else p[0]
        return model, label or (p[2] if len(p) >= 3 else metric), (p[-1] if len(p) >= 4 else ""), unit
    return p[0], label or (p[1] if len(p) > 1 else metric), "", unit


def fmt_val(v, unit):
    if unit == "s":
        return f"{v * 1000:.3g} ms" if v < 1 else f"{v:.3g} s"
    if unit == "mem":
        if v >= 1e9:
            return f"{v / 1e9:.3g} GB"
        return f"{v / 1e6:.3g} MB" if v >= 1e6 else f"{v / 1e3:.3g} kB"
    if unit == "tok":
        return f"{v:.4g} tok/s"
    return f"{v:.4g}"


def cell(metric):
    """two-line table cell: model on top, 'label · variant' below in small text."""
    model, label, variant, _ = describe(metric)
    sub = f"{label} · {variant}" if variant else label
    return f"{model}<br><sub>{sub}</sub>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="dir of per-device result subdirs")
    ap.add_argument("--bench-data", required=True)
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--pr-sha", required=True)
    ap.add_argument("--out", required=True, help="PR comment markdown path")
    args = ap.parse_args()

    cfg = tomllib.load(open(args.thresholds, "rb"))
    min_load = cfg.get("min_load_seconds", 0.0)
    rows, devices, ref_days = [], [], []
    for meta_path in sorted(glob.glob(os.path.join(args.results, "*", "meta.json"))):
        d = os.path.dirname(meta_path)
        meta = json.load(open(meta_path))
        device, triple = meta["device"], meta["triple"]
        devices.append(device)
        pr = read_metrics(os.path.join(d, "metrics"))
        ref, noise, ref_day = reference(args.bench_data, triple, device)
        if ref_day:
            ref_days.append(ref_day)
        for metric, prv in pr.items():
            rv = ref.get(metric)
            if rv is None or rv == 0:
                continue
            delta = (prv - rv) / rv * 100.0
            higher_better = bool(HIGHER_BETTER.search(metric))
            worse = (delta < 0) if higher_better else (delta > 0)
            n = noise.get(metric)
            thr = threshold_for(metric, cfg, n)
            mover = abs(delta) >= thr
            # noisy classes can't be judged on one shot without a measured baseline
            if n is None and any(c in metric for c in cfg.get("needs_history", [])):
                mover = False
            # sub-resolution loads carry no signal; drop them as movers (still shown)
            if "time_to" in metric and rv < min_load:
                mover = False
            rows.append({"device": device, "metric": metric, "ref": rv, "pr": prv,
                         "delta": delta, "worse": worse, "mover": mover})

    movers = [r for r in rows if r["mover"]]
    regr = sorted([r for r in movers if r["worse"]], key=lambda r: -abs(r["delta"]))
    impr = sorted([r for r in movers if not r["worse"]],
                  key=lambda r: (not SPEED.search(r["metric"]), -abs(r["delta"])))

    ref_day = max(ref_days).isoformat() if ref_days else "n/a"
    age = (datetime.date.today() - max(ref_days)).days if ref_days else "?"
    n_metrics = len(rows)
    devs = ", ".join(f"`{d}`" for d in sorted(set(devices)))

    def table(items):
        lines = ["| Δ | metric | device | main → PR |", "|---|---|---|---|"]
        for r in items:
            icon = "🔴" if r["worse"] else "🟢"
            unit = describe(r["metric"])[3]
            lines.append(f"| {icon} **{r['delta']:+.1f}%** | {cell(r['metric'])} "
                         f"| `{r['device']}` | {fmt_val(r['ref'], unit)} → {fmt_val(r['pr'], unit)} |")
        return "\n".join(lines)

    speed_regr = [r for r in regr if SPEED.search(r["metric"])]
    other_regr = [r for r in regr if not SPEED.search(r["metric"])]

    marker = "<!-- bench-vs-main -->"
    if speed_regr:
        head = f"🔴 **Bench vs main — {len(speed_regr)} speed regression(s)**"
        if other_regr:
            head += f" · {len(other_regr)} load/memory"
    elif other_regr:
        head = f"🟡 **Bench vs main — no speed regressions** · {len(other_regr)} load/memory mover(s)"
    else:
        head = "✅ **Bench vs main — no regressions**"
    parts = [marker, head, "",
             f"Reference: main nightly, latest **{ref_day}** ({age}d old) · "
             f"PR `{args.pr_sha[:9]}` · ran on {devs} · {n_metrics} metrics compared", ""]
    # speed is the merge signal — always its own section, with explicit reassurance when clean
    parts += ["**Speed** — evaltime · prefill · decode", ""]
    parts += [table(speed_regr) if speed_regr else "_no inference-speed regressions_", ""]
    if other_regr:
        parts += ["**Load & memory** (worst first)", "", table(other_regr), ""]
    if impr:
        parts += [f"<details><summary>🟢 {len(impr)} improvement(s)</summary>\n\n{table(impr)}\n</details>", ""]
    parts += [f"_lower is better except prefill/decode (tok/s) · adaptive thresholds "
              f"(max(floor, k×noise) vs the series' own history) · single-shot vs nightly "
              f"reference · full table → [run summary]({os.environ.get('RUN_URL', '#')})_"]
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
                    unit = describe(r["metric"])[3]
                    fh.write(f"| {flag} {r['delta']:+.1f}% | {cell(r['metric'])} | "
                             f"{fmt_val(r['ref'], unit)} → {fmt_val(r['pr'], unit)} |\n")
                fh.write("\n")
    print(f"regressions={len(regr)} improvements={len(impr)} metrics={n_metrics}")


if __name__ == "__main__":
    main()
