#!/usr/bin/env python3
"""Translate compacted graphite history -> reviewable bench-data tree.

Reads compact/<platform>__<minion>__<branch>.json (null-stripped real data),
scrubs minion->generic device via the LOCKED mapping, normalizes platform->
build-triple, pivots series by timestamp into one record/run, splits by year,
writes append-style NDJSON: bench-data/<build-triple>/<device>/<year>.ndjson
with lines {"ts":..,"metrics":{name:value}} (commit-less, scalar = historical).

NO stitching (musl/gnu stay separate by build-triple). NO commit (graphite had
none). Writes to scratch only — nothing committed. Refuses unmapped minions.
"""
import json, os, glob, datetime, collections, sys, math

OUT = "/workspace/bench-graphite-export"
DST = f"{OUT}/bench-data"

# machine->generic-device map, excludes, drops come from an untracked config
# (the private mapping table; see recovery-config.example.json).
_CFG = json.load(open(os.environ.get("BENCH_RECOVERY_CONFIG", "recovery-config.json")))
DEVICE = _CFG["device_map"]         # machine id -> generic device name
DROP = set(_CFG["drop_devices"])    # throwaway machines
EXCLUDE = set(_CFG["exclude_devices"])  # never-publish machines

# metric allowlist (leak guard): net/llm model names confirmed PUBLIC (current bench suite +
# retired-public kaldi). Any model NOT listed (e.g. retired private benches) is dropped, unnamed.
ALLOWED_MODELS = {
    "llama_3_1_8B_instruct_q40ef16_541", "llama_3_2_1B_instruct_q40ef16_541",
    "llama_3_2_1B_q40ef32_516", "llama_3_2_3B_instruct_q40ef16_541",
    "llama_3_2_3B_q40ef32_516", "openelm_270M_q40ef16_516", "openelm_270M_q40ef16_541",
    "qwen3_1_7B_q40ef16_541",
    "arm_ml_kws_cnn_m", "dummy_conmer_12M", "en_tdnn_15M", "en_tdnn_15M_nnef", "en_tdnn_8M",
    "en_tdnn_8M_nnef", "en_tdnn_lstm_bn_q7", "en_tdnn_pyt_15M", "hey_snips_v1", "hey_snips_v31",
    "hey_snips_v4_model17", "hey_snips_v4_model17_nnef", "inceptionv1q", "inceptionv3",
    "kaldi_librispeech_clean_tdnn_lstm_1e_256", "mdl_en_2019_Q3_librispeech_onnx",
    "mobilenet_v1_1", "mobilenet_v2_1", "parakeet_tdt_600m_v3_f32f32_decoder_pass",
    "parakeet_tdt_600m_v3_f32f32_encoder_1s", "parakeet_tdt_600m_v3_f32f32_joint_pass",
    "parakeet_tdt_600m_v3_f32f32_preprocessor_1s", "speaker_id", "trunet",
    "voicecom_fake_quant", "voicecom_float",
}
ALLOWED_CATS = {"binary_size", "bundle"}   # non-model public operational metrics
_dropped = collections.Counter()

def allowed(metric):
    parts = metric.split(".")
    cat = parts[0]
    if cat in ("net", "llm"):
        ok = len(parts) > 1 and parts[1] in ALLOWED_MODELS
    else:
        ok = cat in ALLOWED_CATS
    if not ok:
        _dropped[cat if cat not in ("net", "llm") else f"{cat}.{parts[1] if len(parts)>1 else '?'}"] += 1
    return ok

def build_triple(platform):
    return platform.replace("_", "-")   # raspbian stays raspbian

def sig(x, n=4):
    """round to n significant figures; drop .0 when integral (kills false precision)."""
    if x is None or x == 0:
        return x
    r = round(x, -int(math.floor(math.log10(abs(x)))) + (n - 1))
    return int(r) if r == int(r) else r

def day_of(ts):
    return datetime.datetime.fromtimestamp(ts, datetime.UTC).date()

def main():
    if os.path.exists(DST):
        import shutil; shutil.rmtree(DST)
    written = collections.defaultdict(int)   # (triple,device) -> records
    skipped = []
    for f in sorted(glob.glob(f"{OUT}/compact/*.json")):
        platform, minion, branch = os.path.basename(f)[:-5].split("__")
        if minion in EXCLUDE or minion in DROP:
            skipped.append((minion, "dropped/excluded")); continue
        if minion not in DEVICE:
            print(f"  !! UNMAPPED minion {minion} — refusing", file=sys.stderr)
            sys.exit(2)
        device = DEVICE[minion]; triple = build_triple(platform)
        series = json.load(open(f))
        # pivot to DAY granularity: date -> {metric: rounded value}  (last wins intra-day)
        byday = collections.defaultdict(dict)
        for s in series:
            parts = s["target"].split(".")
            metric = ".".join(parts[4:])      # after tract.<platform>.<minion>.<branch>
            if not allowed(metric):
                continue
            for v, ts in s["points"]:
                byday[day_of(ts)][metric] = sig(v)
        # one COLUMNAR file per series: start_day + one slot/day (null if no bench that day);
        # column i == start_day + i days. arrays all same length = span.
        days = sorted(byday)
        start, end = days[0], days[-1]
        span = [start + datetime.timedelta(d) for d in range((end - start).days + 1)]
        mset = sorted({m for dd in days for m in byday[dd]})
        obj = {"start_day": start.isoformat(),
               "metrics": {m: [byday.get(day, {}).get(m) for day in span] for m in mset}}
        os.makedirs(f"{DST}/{triple}", exist_ok=True)
        with open(f"{DST}/{triple}/{device}.json", "w") as fh:
            json.dump(obj, fh, separators=(',', ':'))
        written[(triple, device)] = (len(span), len(days))   # (grid days, actual bench days)

    print(f"{'build-triple / device':62} {'grid':>5} {'bench':>5} {'null%':>5}")
    for (t, dev), (grid, actual) in sorted(written.items()):
        print(f"  {t+'/'+dev:60} {grid:5} {actual:5} {100*(grid-actual)//grid:4}%")
    print(f"\nskipped devices: {skipped}")
    print(f"DROPPED metrics (leak guard): {dict(_dropped)}")
    print(f"tree at {DST}")

if __name__ == "__main__":
    main()
