# tract bench backfill — locked era-builds (2017→2024)

Reproducibility bundle for benching historical tract releases on today's infra.
Each entry rebuilds a past tract, built faithfully with its **contemporaneous compiler
and its contemporaneous crates.io dependency graph**, so the numbers form a
self-consistent curve (see `curve.csv`). Method: resolve against a git snapshot of
`rust-lang/crates.io-index-archive` at the tag's date, freeze the result as `Cargo.lock`,
build with the era `rustc`. Engine source is **never** modified — patches touch only
build scaffolding (workspace trims, build-script/dep ablations, a lib timing harness).

## Layout
- `manifest.tsv` — one row per era-point: id, date, base_ref, rustc, index_archive_branch,
  index_commit, run_recipe, result.
- `entries/<id>/Cargo.lock` — the frozen era dependency graph (THE lock).
- `entries/<id>/modernization.patch` — build-scaffolding edits vs the base tag (may be empty).
- `entries/<id>/base_commit.txt` — exact base commit SHA.
- `scripts/` — the driver scripts that produced the builds (exact features + run args).
- `curve.csv` — the assembled results.

## Rebuild one entry
```sh
id=2021-06                                   # pick from manifest.tsv
base=$(cat entries/$id/base_commit.txt)
rustc=$(awk -F'\t' -v i=$id '$1==i{print $4}' manifest.tsv)
rustup toolchain install "$rustc" --profile minimal
git -C $TRACT worktree add --detach /tmp/rb-$id "$base"
[ -s entries/$id/modernization.patch ] && git -C /tmp/rb-$id apply "$PWD/entries/$id/modernization.patch"
cp entries/$id/Cargo.lock /tmp/rb-$id/Cargo.lock
cd /tmp/rb-$id && CARGO_TARGET_DIR=/tmp/rb-$id-tgt rustup run "$rustc" cargo build --release --locked
# then run per manifest.tsv run_recipe (CLI syntax differs by era: -s WxHxf32 + profile pre-2019,
# -O bench 2020+, -i uses 'x' sep <=0.12 and commas later; 2017 = --example inceptionv3)
```
`--locked` fetches the exact pinned crates from crates.io (all historical versions persist),
so **the multi-GB index registries are
NOT needed for rebuild and can be deleted** — they were only used to *generate* the locks.

## If --locked can't fetch (crates.io git index unavailable to old cargo)
Re-materialize the era index from the archive and use source replacement:
```sh
sha=$(awk -F'\t' -v i=$id '$1==i{print $6}' manifest.tsv)         # index_commit
gh api repos/rust-lang/crates.io-index-archive/tarball/$sha > idx.tgz
mkdir idx && tar xzf idx.tgz -C idx --strip-components=1
( cd idx && git init -q && git add -A && git commit -qm snap )
# in the worktree .cargo/  (BOTH names: cargo <1.39 reads `config`, >=1.39 reads `config.toml`):
printf '[source.crates-io]\nreplace-with="s"\n[source.s]\nregistry="file://%s/idx"\n' "$PWD" \
  | tee /tmp/rb-$id/.cargo/config /tmp/rb-$id/.cargo/config.toml
```

## Future: a53 (and other board) cross-builds
The locks are architecture-independent. To bench these era-builds on the cortex-a53 (etc.):
build with the same (base + patch + Cargo.lock + era rustc) but add
`--target aarch64-unknown-linux-musl` and an era-compatible cross toolchain (cross.sh style:
gnu cross-gcc for tract-linalg's cc build + musl linker), then run on the board via dinghy
(`cargo-dinghy -d <alias> runner ./tract -- ...`). Model files stream/served as in the live
fleet. NOTE: very old `rustc` may lack a prebuilt `aarch64-unknown-linux-musl` std — install
via `rustup target add` per toolchain; if unavailable for the era rustc, that entry's a53
number needs a slightly newer rustc (document the deviation).

## Caveats baked into the numbers
- **2017-09** is a COLD SINGLE inference via the pre-1.0 lib API (it caches node values across
  `run()`, so no steady-state loop was possible) — an upper bound; later points are warmed
  `-O bench`/`profile --bench` steady-state means. Not level-identical methodology.
- Era-compiler varies (rustc 1.31→1.81): compiler codegen gains are folded in ("what shipped").
- Ablations (git2/onnx build-script, conform/libtensorflow, reqwest/openssl download, example
  dev-deps) are build-scaffolding only — inference engine untouched, so timings stay attributable.
