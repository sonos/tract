# Profile-guided optimization (PGO) and BOLT

tract spends almost all of its runtime in a small amount of hot code: the
`tract-linalg` matmul micro-kernels and the per-op dispatch/glue that drives
them. Two post-profile build steps can squeeze a few percent more out of a
release binary without touching the source:

* **PGO** (`-Cprofile-generate` / `-Cprofile-use`) lets LLVM lay out branches,
  inline, and order code using counts gathered from a representative run.
* **BOLT** (`llvm-bolt`) re-optimizes the *linked* binary's code layout
  (block/function reordering, hot/cold splitting) from a second profile, on top
  of PGO.

Both rely on a **training run** over a basket of models, and both produce
profiles that are tied to one architecture and one toolchain — see
[Portability](#portability) before committing anything.

This is a release-engineering recipe, not part of the normal build. The
profiles are regenerated per target; tract does not ship profile blobs.

## Prerequisites

* `llvm-profdata` **matching rustc's LLVM** — install it as a rustup
  component so the version always lines up with the compiler:

  ```bash
  rustup component add llvm-tools-preview
  PROFDATA=$(find "$(rustc --print sysroot)" -name llvm-profdata | head -1)
  ```

  The system `llvm-profdata` from a distro LLVM will usually be a different
  version and reject the `.profraw` files.

* For BOLT: `llvm-bolt` and `merge-fdata` (e.g. Ubuntu's `llvm-bolt`
  package). The instrumentation runtime (`libbolt_rt_instr.a`) ships next to
  the versioned binary, so prefer the versioned path if the `llvm-bolt`
  wrapper can't find it:

  ```bash
  BOLT=/usr/lib/llvm-18/bin/llvm-bolt
  MERGE=/usr/lib/llvm-18/bin/merge-fdata
  ```

## The training basket

The profile is only as good as what you run through it. Pick models that
exercise the regimes tract actually hits, and include controls so you can tell
real wins from noise. A workable basket spans three regimes:

| regime              | example models                              |
|---------------------|---------------------------------------------|
| branchy / dispatch-heavy CNNs | inception_v3, mobilenet_v2, squeezenet, yolov8n |
| memory-bound streaming        | an encoder / streaming model                |
| pure GEMM (controls)          | square `matmul`, skinny `matmul`            |

A pure square GEMM should barely move — it is almost all time in one
hand-written kernel that neither PGO nor BOLT reshapes much. If a control
swings wildly, your measurement is too noisy (small kernels finish in
microseconds; pin to one core and take a best-of-N).

`bench` loops internally, so **one invocation per model** under
`-O bench --allow-random-input` already yields plenty of profile data.

## PGO

```bash
FEATURES="--no-default-features --features onnx,tf"   # the CPU features you ship
RAW=$PWD/target/pgo-raw
rm -rf "$RAW"; mkdir -p "$RAW"

# 1. instrumented build
RUSTFLAGS="-Cprofile-generate=$RAW" \
  cargo build --release -p tract-cli $FEATURES

# 2. train: one bench run per model, each to its own .profraw
for m in "${BASKET[@]}"; do
  LLVM_PROFILE_FILE="$RAW/$(basename "$m")-%p.profraw" \
    ./target/release/tract "$m" -O --machine-friendly \
      bench --allow-random-input >/dev/null
done

# 3. merge (with the rustc-matched llvm-profdata)
$PROFDATA merge -o "$PWD/target/basket.profdata" "$RAW"/*.profraw

# 4. rebuild using the profile
RUSTFLAGS="-Cprofile-use=$PWD/target/basket.profdata -Cllvm-args=-pgo-warn-missing-function" \
  cargo build --release -p tract-cli $FEATURES
```

`-pgo-warn-missing-function` turns stale-profile mismatches into warnings
instead of letting them pass silently — useful when the source has drifted
from the profile.

## BOLT (on top of PGO)

BOLT rewrites the linked binary, so build the PGO binary with relocations
retained (`-Wl,--emit-relocs`); this does not change codegen, only what
survives in the ELF:

```bash
RUSTFLAGS="-Cprofile-use=$PWD/target/basket.profdata -Clink-args=-Wl,--emit-relocs" \
  cargo build --release -p tract-cli $FEATURES
cp target/release/tract tract.pgo
```

### The asm-kernel caveat

tract's `tract-linalg` micro-kernels are hand-written assembly emitted with
the crate version baked into each symbol, e.g.
`avx512_mmm_f32_16x12_0_23_2_pre`. BOLT's disassembler cannot follow their
control flow and aborts with `disassembly failed - inconsistent branch found`.
Skip them with a regex on the version suffix (match your crate version):

```bash
SKIP='.*_0_23_2_pre.*'
```

BOLT then optimizes everything *except* the GEMM kernels — i.e. the op
dispatch, conv glue, im2col, and reductions — which is exactly the branchy
code where layout matters most. The kernels are already optimal asm, so
skipping them costs little.

### Instrument, train, optimize

```bash
FD=$PWD/bolt-fdata; rm -rf "$FD"; mkdir -p "$FD"

# instrument
$BOLT tract.pgo -instrument \
  -instrumentation-file="$FD/prof.fdata" -instrumentation-file-append-pid \
  -skip-funcs="$SKIP" \
  -o tract.bolt.inst

# train on the same basket (one bench run each)
for m in "${BASKET[@]}"; do
  ./tract.bolt.inst "$m" -O --machine-friendly \
    bench --allow-random-input >/dev/null
done

# merge per-process profiles, then optimize
$MERGE "$FD"/*.fdata > bolt.merged.fdata
$BOLT tract.pgo -data=bolt.merged.fdata \
  -skip-funcs="$SKIP" \
  -reorder-blocks=ext-tsp -reorder-functions=hfsort \
  -split-functions -split-all-cold -split-eh -icf=1 \
  -dyno-stats \
  -o tract.bolt
```

`-dyno-stats` prints a before/after summary (e.g. executed forward branches)
so you can sanity-check that BOLT did something before benchmarking.

## Measuring

Benchmark `target/release/tract` (baseline), `tract.pgo`, and `tract.bolt`
the same way — pinned to one core, best-of-N, machine-friendly output:

```bash
taskset -c 0 ./tract.bolt "$model" -O --machine-friendly \
  bench --allow-random-input --max-time 4000 --warmup-time 1000
```

### Results

<!-- Replace with the canonical per-target numbers measured for the MR. -->
Illustrative, x86_64 (4 cores), single-thread, best-of-5:

| model              | baseline | PGO    | PGO+BOLT |
|--------------------|---------:|-------:|---------:|
| inception_v3       |   +0.0%  | +0.2%  |  +6.5%   |
| mobilenet_v2       |   +0.0%  | +6.7%  |  +7.7%   |
| squeezenet         |   +0.0%  | +2.0%  |  +5.5%   |
| yolov8n            |   +0.0%  | +6.3%  | +10.6%   |
| matmul (square)    |   +0.0%  | ~noise |  ~noise  |
| matmul (skinny)    |   +0.0%  | +2.0%  |  +3.9%   |

BOLT is additive with PGO and helps the branchy CNNs most — consistent with
its gains coming from layout of the dispatch/runtime code that is left after
the asm kernels are skipped.

## Portability

Neither profile is a portable artifact, which is why they are not checked in:

* **PGO `.profdata`** is collected at the LLVM-IR level, so it is keyed by
  function name + a structural hash of each function. A different rustc/LLVM
  version or source drift invalidates entries (they are dropped with a
  warning). More importantly, tract's hottest code is `cfg(target_arch)`-gated:
  a profile gathered on x86 has *zero* coverage of the arm64 NEON/SDOT kernels
  and vice-versa, so the part that matters most is uncovered on the other arch.

* **BOLT `.fdata`** references concrete symbols/addresses of one specific
  binary and is invalidated by any rebuild, let alone a different ISA.

The reusable, architecture-independent part is this recipe and the basket.
Wire them into the per-target release build so each artifact regenerates its
own profiles — the same pattern rustc's own `opt-dist`, Firefox, and Clang
use — rather than committing profile blobs that go stale on every toolchain
bump.
