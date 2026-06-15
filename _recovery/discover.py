#!/usr/bin/env python3
"""Discovery pass over tract bench history in Graphite (fast globstar method).

Per platform: enumerate machine ids.
Per (platform, machine, branch in main/master): ONE globstar render
(tract.p.m.branch.**) to get series count + actual data span. Light maxDataPoints
so this stays a discovery pass, not the bulk pull. Read-only; throttled;
checkpointed; writes only to this scratch dir (outside any repo).

Goal: produce the machine-id list + the main-vs-master spans, before pulling
any real data. Endpoint/prefix come from an untracked config (see
recovery-config.example.json); set BENCH_RECOVERY_CONFIG to point elsewhere.
"""
import json, os, ssl, sys, time, urllib.parse, urllib.request

_CFG = json.load(open(os.environ.get("BENCH_RECOVERY_CONFIG", "recovery-config.json")))
BASE = _CFG["graphite_host"]
ROOT = _CFG["graphite_prefix"]
BRANCHES = ["main", "master"]
FROM = "-8years"
MAXPTS = "400"            # ~weekly over 8y: enough to locate span, small response
THROTTLE_S = 1.0
TIMEOUT_S = 90            # host is brutally slow
RETRIES = 3
OUT = "/workspace/bench-graphite-export"
CTX = ssl.create_default_context()

def fetch(path, params):
    url = f"{BASE}{path}?" + urllib.parse.urlencode(params)
    last = None
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT_S, context=CTX) as r:
                data = r.read()
            time.sleep(THROTTLE_S)
            return json.loads(data)
        except Exception as e:                       # noqa: BLE001
            last = e
            time.sleep(3 * (attempt + 1))
    print(f"  ! giving up: {path} {params.get('query') or params.get('target')}: {last}",
          file=sys.stderr)
    return None

def span(series):
    """(first_ts, last_ts, total_nonnull) across all series datapoints."""
    first = last = None
    total = 0
    for s in series:
        nn = [p for p in s.get("datapoints", []) if p[0] is not None]
        total += len(nn)
        if nn:
            first = nn[0][1] if first is None else min(first, nn[0][1])
            last = nn[-1][1] if last is None else max(last, nn[-1][1])
    return first, last, total

def main():
    disco = {}
    platforms = [n["text"] for n in (fetch("/metrics/find",
                 {"query": f"{ROOT}.*", "format": "treejson"}) or [])]
    print(f"platforms ({len(platforms)}): {platforms}\n")
    for p in platforms:
        minions = [n["text"] for n in (fetch("/metrics/find",
                   {"query": f"{ROOT}.{p}.*", "format": "treejson"}) or [])]
        disco[p] = {}
        print(f"{p}: minions = {minions}")
        for m in minions:
            disco[p][m] = {}
            for b in BRANCHES:
                series = fetch("/render", {"target": f"{ROOT}.{p}.{m}.{b}.**",
                               "format": "json", "from": FROM, "until": "now",
                               "maxDataPoints": MAXPTS})
                if series:
                    f0, l0, tot = span(series)
                    disco[p][m][b] = {"series": len(series), "first_ts": f0,
                                      "last_ts": l0, "nonnull": tot}
                    print(f"    {m}.{b}: {len(series)} series, span {f0}..{l0}")
                else:
                    disco[p][m][b] = None
            with open(f"{OUT}/discovery.json", "w") as fh:
                json.dump(disco, fh, indent=2)

    # codename list (union of all minion-ids) for the scrub mapping
    codenames = sorted({m for p in disco for m in disco[p]})
    with open(f"{OUT}/codenames.json", "w") as fh:
        json.dump(codenames, fh, indent=2)
    print(f"\nminion-ids (codenames to scrub): {codenames}")
    print("DONE")

if __name__ == "__main__":
    main()
