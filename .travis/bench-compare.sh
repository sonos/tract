#!/usr/bin/env bash
# .travis/bench-compare.sh <ref-a> <ref-b>
#
# Build tract for two git references, run bundle-entrypoint.sh for each,
# and print a side-by-side evaltime comparison.
#
# Only net.*.evaltime.* and llm.*.(pp|tg)*.* metrics are shown.
# Forwarded env vars:
#   CACHEDIR    model cache directory (default: ~/.cache/tract-ci-minion-models)
#   BENCH_OPTS  extra bench flags passed through to tract

set -euo pipefail

REF_A=${1:?'usage: bench-compare.sh <ref-a> <ref-b>'}
REF_B=${2:?'usage: bench-compare.sh <ref-a> <ref-b>'}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
CACHEDIR=${CACHEDIR:-$HOME/.cache/tract-ci-minion-models}

resolve_ref() {
    local ref=$1
    [[ $ref =~ ^[0-9]{4}$ ]] && ref="refs/pull/$ref/head"
    local sha
    sha=$(git -C "$REPO_ROOT" rev-parse --verify "$ref^{commit}" 2>/dev/null) \
        && { echo "$sha"; return; }
    if [[ $ref == refs/pull/*/head || $ref == refs/pull/*/merge ]]; then
        echo "==> fetching $ref" >&2
        git -C "$REPO_ROOT" fetch origin "$ref:$ref" >&2 \
            || { echo "error: could not fetch '$ref' from origin" >&2; return 1; }
        sha=$(git -C "$REPO_ROOT" rev-parse --verify "$ref^{commit}" 2>/dev/null) \
            && { echo "$sha"; return; }
    fi
    echo "error: '$ref' does not resolve to a commit" >&2
    return 1
}
ok=1
COMMIT_A=$(resolve_ref "$REF_A") || ok=0
COMMIT_B=$(resolve_ref "$REF_B") || ok=0
[ $ok -eq 1 ] || exit 1

WORK=$(mktemp -d /tmp/bench-compare.XXXXXXXX)

cleanup() {
    git -C "$REPO_ROOT" worktree remove --force "$WORK/wt_a" 2>/dev/null || true
    git -C "$REPO_ROOT" worktree remove --force "$WORK/wt_b" 2>/dev/null || true
    rm -rf "$WORK"
}
trap cleanup EXIT

# ── build ─────────────────────────────────────────────────────────────────────

build_ref() {
    local label=$1 ref=$2
    local wt="$WORK/wt_$label"
    echo "==> worktree $ref" >&2
    git -C "$REPO_ROOT" worktree add --detach "$wt" "$ref" >&2
    echo "==> build $ref" >&2
    cargo build --manifest-path "$wt/Cargo.toml" \
        -p tract-cli -q --release \
        --target-dir "$WORK/target_$label" >&2
    echo "$WORK/target_$label/release/tract"
}

TRACT_A=$(build_ref a "$COMMIT_A")
TRACT_B=$(build_ref b "$COMMIT_B")

# ── bench ─────────────────────────────────────────────────────────────────────

run_bench() {
    local label=$1 tract=$2
    local rundir="$WORK/run_$label"
    mkdir -p "$rundir"
    echo "==> bench [$label] $REF_A / $REF_B" >&2
    (
        cd "$rundir"
        export TRACT_RUN="$tract"
        export CACHEDIR="$CACHEDIR"
        [ -n "${BENCH_OPTS:-}" ] && export BENCH_OPTS
        bash "$SCRIPT_DIR/bundle-entrypoint.sh"
    ) 2>&1 | sed "s/^/  [$label] /" >&2
    if [ ! -f "$rundir/metrics" ]; then
        echo "error: bench run failed for '$label' — no metrics produced" >&2
        return 1
    fi
    echo "$rundir/metrics"
}

METRICS_A=$(run_bench a "$TRACT_A")
METRICS_B=$(run_bench b "$TRACT_B")

# ── comparison table ──────────────────────────────────────────────────────────

filter_eval() {
    # evaltime (ns), pp* and tg* (llm tokens/s or ms/token)
    grep -E '\.(evaltime|pp[0-9]+|tg[0-9]+)\.' "$1" || true
}

declare -A va vb
while read -r k v; do va[$k]=$v; done < <(filter_eval "$METRICS_A")
while read -r k v; do vb[$k]=$v; done < <(filter_eval "$METRICS_B")

all_keys=$(
    { filter_eval "$METRICS_A"; filter_eval "$METRICS_B"; } \
        | awk '{print $1}' | sort -u
)

if [ -t 1 ]; then
    RED='\033[0;31m' GRN='\033[0;32m' YEL='\033[1;33m' RST='\033[0m'
else
    RED='' GRN='' YEL='' RST=''
fi

hrule() { printf '%.0s-' $(seq 1 "$1"); }

W=62
printf "\n"
printf "  %-${W}s  %13s  %13s  %8s  %7s\n" "metric" "$REF_A" "$REF_B" "ratio" "delta%"
printf "  %-${W}s  %13s  %13s  %8s  %7s\n" \
    "$(hrule $W)" "$(hrule 13)" "$(hrule 13)" "$(hrule 8)" "$(hrule 7)"

while IFS= read -r key; do
    [[ -z $key ]] && continue
    a_val=${va[$key]:-}
    b_val=${vb[$key]:-}
    if [[ -n $a_val && -n $b_val ]]; then
        read -r ratio delta colour < <(awk -v a="$a_val" -v b="$b_val" 'BEGIN {
            if (a == 0) { print "N/A N/A YEL"; exit }
            r = b / a
            c = (r > 1.05) ? "RED" : (r < 0.95) ? "GRN" : "RST"
            printf "%.4f %+.1f %s\n", r, (r-1)*100, c
        }')
        case $colour in
            RED) c=$RED ;;
            GRN) c=$GRN ;;
            YEL) printf "  ${YEL}%-${W}s  %13s  %13s  %8s  %7s${RST}\n" \
                     "$key" "$a_val" "$b_val" "N/A" "N/A"; continue ;;
            *)   c=$RST ;;
        esac
        printf "  ${c}%-${W}s  %13s  %13s  %8s  %6s%%${RST}\n" \
            "$key" "$a_val" "$b_val" "$ratio" "$delta"
    else
        printf "  ${YEL}%-${W}s  %13s  %13s  %8s  %7s${RST}\n" \
            "$key" "${a_val:-N/A}" "${b_val:-N/A}" "N/A" "N/A"
    fi
done <<< "$all_keys"

printf "\n  evaltime in ns; llm pp/tg units from tract llm-bench output\n"
printf "  green = REF_B faster (ratio < 0.95)   red = REF_B slower (ratio > 1.05)\n\n"
