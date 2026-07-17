#!/usr/bin/env bash
# Compare the end-to-end bench-suite between two commits on THIS machine — the local
# equivalent of the CI bench comment, for cohorts CI doesn't bench (your own PCs/Macs).
# Builds tract-cli --features bench-suite at each ref, runs the model battery from
# benches.toml, and prints the per-metric evaltime delta of B relative to A via the
# `bench-diff` subcommand. Anything after the two refs is forwarded to bench-suite
# (e.g. --filter en_tdnn to scope the battery).
#
#   .travis/bench-compare.sh <ref-a> <ref-b> [bench-suite args...]
#   .travis/bench-compare.sh main task/mmm-restream-term --filter en_tdnn
#
# B must carry the `bench-diff` subcommand (it renders the comparison) — so pass the newer
# commit as B. Checks each ref out in place and restores the original on exit, so the working
# tree must be clean. Builds --release to match the CI bench profile; the comparison is
# relative, so absolute numbers vary with the machine's thermal/governor state — pin
# performance for stable runs, but back-to-back A-vs-B on one box is robust regardless.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

a="${1:?usage: bench-compare.sh <ref-a> <ref-b> [bench-suite args...]}"
b="${2:?usage: bench-compare.sh <ref-a> <ref-b> [bench-suite args...]}"
shift 2

git diff-index --quiet HEAD -- || { echo "working tree is dirty; commit or stash first" >&2; exit 1; }

orig="$(git symbolic-ref --quiet --short HEAD || git rev-parse HEAD)"
out="$(mktemp -d)"
trap 'git checkout --quiet "$orig"' EXIT

slug() { echo "$1" | tr '/ :' '___'; }

run() {
  local ref="$1"; shift
  echo ">> $ref: building tract-cli --features bench-suite" >&2
  git checkout --quiet "$ref"
  cargo build --release -p tract-cli --features bench-suite >&2
  echo ">> $ref: running bench-suite" >&2
  target/release/tract bench-suite --manifest .travis/benches.toml --output "$out/$(slug "$ref")" "$@"
}

run "$a" "$@"
run "$b" "$@"   # leaves the tree on B, whose binary renders the diff

echo
target/release/tract bench-diff --a "$out/$(slug "$a")" --b "$out/$(slug "$b")"
echo
echo "metrics: $out/$(slug "$a") , $out/$(slug "$b")" >&2
echo "re-view: tract bench-diff --a <A> --b <B> [--metric load|rss] [--threshold 5]" >&2
