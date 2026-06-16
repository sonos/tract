"""Shared bench-comparison logic.

The one place that answers "what move makes this metric a PR red?" — used by both
bench-report.py (to flag reds) and bench-expectations.py (so the bundle retries
exactly the would-be reds). Keeping it single-sourced is what guarantees that every
red the maintainer sees was measured more than once.
"""
import re, tomllib

HIGHER_BETTER = re.compile(r"\.(pp\d+|tg\d+)\.")   # llm throughput; everything else lower-is-better


def load_cfg(path):
    return tomllib.load(open(path, "rb"))


def read_metrics(path):
    out = {}
    for line in open(path):
        p = line.split()
        if len(p) >= 2:
            try:
                # '-' -> '_' to match the recovered history and nightly references
                # (the old minion ran metric names through `tr '-' '_'` for graphite).
                out[p[0].replace("-", "_")] = float(p[1])
            except ValueError:
                pass
    return out


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


def floor_for(metric, floors):
    for cls, t in floors.items():
        if cls != "default" and cls in metric:
            return t
    return floors.get("default", 5)


def red_threshold(metric, cfg, noise, value):
    """The |Δ%| that makes this metric a PR red, or None if it is never gated:
    operational/ignored, a sub-resolution load, or a noisy class lacking the history
    to estimate its noise. Otherwise max(class floor, k * the series' own noise)."""
    if any(c in metric for c in cfg.get("ignore", [])):
        return None
    if "time_to" in metric and value is not None and value < cfg.get("min_load_seconds", 0):
        return None
    if noise is None and any(c in metric for c in cfg.get("needs_history", [])):
        return None
    floor = floor_for(metric, cfg["floors"])
    return floor if noise is None else max(floor, cfg.get("k", 3.0) * noise)
