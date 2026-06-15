#!/usr/bin/env python3
"""Raw full-resolution capture of recoverable tract bench history from Graphite.

Insurance against the retention horizon: freezes a LOCAL private snapshot of every
real-data series before more of it rolls off. NO scrubbing, NO naming, NO stitching,
NO merging musl/gnu — that's all downstream transform against this copy. Build-config
(platform name) is preserved verbatim because graphite already keys by it.

- Drives off discovery.json: captures every (platform, machine, branch) that has data.
- Excludes machines listed in the config (never recover/publish).
- Full resolution (high maxDataPoints), hardier retry for the slow 500/504-prone host.
- Checkpoint/resume: skips files already written, so it survives interruption.
- Writes raw render JSON per series-group to ./raw/, plus a manifest. Private scratch.
Endpoint/prefix/excludes come from an untracked config (recovery-config.example.json).
"""
import json, os, ssl, sys, time, urllib.parse, urllib.request

_CFG = json.load(open(os.environ.get("BENCH_RECOVERY_CONFIG", "recovery-config.json")))
BASE = _CFG["graphite_host"]
ROOT = _CFG["graphite_prefix"]
EXCLUDE_MINIONS = set(_CFG["exclude_devices"])
FROM = "-4years"                  # covers 2023-03..now w/ margin
MAXPTS = "20000"                  # preserve native (~daily) resolution
THROTTLE_S = 1.0
TIMEOUT_S = 120                   # slow host
RETRIES = 6                       # 500/504 are transient overload
OUT = "/workspace/bench-graphite-export"
RAW = f"{OUT}/raw"
CTX = ssl.create_default_context()

def get(path, params):
    url = f"{BASE}{path}?" + urllib.parse.urlencode(params)
    last = None
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT_S, context=CTX) as r:
                data = r.read()
            time.sleep(THROTTLE_S)
            return data
        except Exception as e:                       # noqa: BLE001
            last = e
            time.sleep(min(60, 4 * (attempt + 1)))   # linear backoff, capped
    print(f"  ! FAILED {params.get('target')}: {last}", file=sys.stderr)
    return None

def main():
    os.makedirs(RAW, exist_ok=True)
    disco = json.load(open(f"{OUT}/discovery.json"))
    # build worklist: (platform, minion, branch) with real data, minus excludes
    work = []
    for p in disco:
        for m in disco[p]:
            if m in EXCLUDE_MINIONS:
                continue
            for b, info in disco[p][m].items():
                if info and info.get("first_ts"):     # has retrievable data
                    work.append((p, m, b))
    print(f"worklist: {len(work)} series-groups (excluded: {EXCLUDE_MINIONS})")
    manifest = []
    for i, (p, m, b) in enumerate(work, 1):
        fname = f"{RAW}/{p}__{m}__{b}.json"
        if os.path.exists(fname) and os.path.getsize(fname) > 2:
            print(f"  [{i}/{len(work)}] skip (have) {p}/{m}/{b}")
            manifest.append({"platform": p, "minion": m, "branch": b, "file": fname, "status": "cached"})
            continue
        print(f"  [{i}/{len(work)}] pull {p}/{m}/{b}")
        data = get("/render", {"target": f"{ROOT}.{p}.{m}.{b}.**",
                              "format": "json", "from": FROM, "until": "now",
                              "maxDataPoints": MAXPTS})
        if data is None:
            manifest.append({"platform": p, "minion": m, "branch": b, "status": "FAILED"})
        else:
            with open(fname, "wb") as fh:
                fh.write(data)
            try:
                n = len(json.loads(data))
            except Exception:
                n = "?"
            manifest.append({"platform": p, "minion": m, "branch": b, "file": fname,
                             "series": n, "bytes": len(data), "status": "ok"})
        with open(f"{OUT}/capture_manifest.json", "w") as fh:
            json.dump(manifest, fh, indent=2)
    ok = sum(1 for x in manifest if x["status"] in ("ok", "cached"))
    print(f"\nDONE: {ok}/{len(work)} captured. raw/ + capture_manifest.json")

if __name__ == "__main__":
    main()
