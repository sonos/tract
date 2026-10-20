#!/usr/bin/env python3
"""Render the README's benchmark sparklines from the bench-data branch.

Each line is relative speed: a 7-point rolling median of the nightly series,
divided into (latency) or by (throughput) the median of its first 7 points, so
every line rises when tract gets faster. Writes one SVG per entry next to this
script and prints the matching README table rows.

    python3 doc/bench/sparklines.py [--data <bench-data checkout or URL>]
"""

import argparse
import datetime
import json
import os
import statistics
import urllib.request

SERIES = [
    ("gtcrn-cortex-a55", "GTCRN speech enhancement", "Cortex-A55",
     "aarch64-unknown-linux-gnu-stretch/cortex-a55", "net.gtcrn.evaltime.frame"),
    ("mobilenet-v2-cortex-a53", "MobileNet v2, multi-thread", "Cortex-A53",
     "aarch64-unknown-linux-gnu-stretch/cortex-a53", "net.mobilenet_v2_1.evaltime.pass_mt"),
    ("parakeet-i9-11900kb", "Parakeet TDT 0.6B preprocessor", "Intel i9-11900KB",
     "x86_64-unknown-linux-gnu/i9-11900kb_rtx-4060", "net.parakeet_tdt_600m_v3_f32f32_preprocessor_1s.evaltime.cpu"),
    ("llama-3b-m1-metal", "Llama 3.2 3B text generation", "Apple M1 Max, Metal",
     "aarch64-apple-darwin/apple-m1-max", "llm.llama_3_2_3B_q40ef32_516.tg128.metal"),
]
W, H, PAD = 240, 48, 4


def load(data, path):
    src = f"{data}/{path}.json"
    if src.startswith("http"):
        with urllib.request.urlopen(src) as r:
            return json.load(r)
    with open(src) as f:
        return json.load(f)


def speed(doc, metric):
    start = datetime.date.fromisoformat(doc["start_day"])
    pts = [(start + datetime.timedelta(i), v) for i, v in enumerate(doc["metrics"][metric]) if v]
    smooth = [(d, statistics.median(v for _, v in pts[max(0, i - 6):i + 1])) for i, (d, _) in enumerate(pts)]
    base = statistics.median(v for _, v in pts[:7])
    latency = ".evaltime." in metric or ".time_to_" in metric
    return [(d, base / v if latency else v / base) for d, v in smooth]


def svg(title, pts):
    d0, d1 = pts[0][0], pts[-1][0]
    hi = max(v for _, v in pts) * 1.05
    lo = min(min(v for _, v in pts), 1.0) * 0.95
    x = lambda d: PAD + (d - d0).days / max(1, (d1 - d0).days) * (W - 2 * PAD)
    y = lambda v: PAD + (hi - v) / (hi - lo) * (H - 2 * PAD)
    line = " ".join(f"{'M' if i == 0 else 'L'}{x(d):.1f} {y(v):.1f}" for i, (d, v) in enumerate(pts))
    ex, ey = x(d1), y(pts[-1][1])
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img">
<title>{title}</title>
<style>.base{{stroke:#9aa3ad}} .line{{stroke:#2a78d6}} .dot{{fill:#2a78d6}}
@media (prefers-color-scheme: dark){{.base{{stroke:#5d656f}} .line{{stroke:#3987e5}} .dot{{fill:#3987e5}}}}</style>
<line class="base" x1="{PAD}" x2="{W - PAD}" y1="{y(1.0):.1f}" y2="{y(1.0):.1f}" stroke-width="1" stroke-dasharray="3 3"/>
<path class="line" d="{line}" fill="none" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>
<circle class="dot" cx="{ex:.1f}" cy="{ey:.1f}" r="3"/>
</svg>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="https://raw.githubusercontent.com/sonos/tract/bench-data")
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    for name, model, target, path, metric in SERIES:
        pts = speed(load(args.data, path), metric)
        gain = pts[-1][1]
        with open(os.path.join(here, f"{name}.svg"), "w") as f:
            f.write(svg(f"{model} on {target}: {gain:.1f}x faster", pts))
        since = pts[0][0].strftime("%b %Y")
        print(f"| {model} | {target} | **{gain:.1f}×** since {since} | ![{model} on {target}](doc/bench/{name}.svg) |")


if __name__ == "__main__":
    main()
