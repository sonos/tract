# Profile-guided optimization (PGO)

`tract`'s release profile already enables LTO. On top of that, a
profile-guided build can give the engine glue — the plan executor, op
dispatch, and tensor bookkeeping that every model runs through — a layout the
compiler tuned to your actual workload. The hand-written `tract-linalg`
matmul kernels are not affected: PGO does not instrument assembly, so the win
concentrates in dispatch-heavy graphs (many small ops) and all but vanishes
for graphs dominated by a steady-state matmul.

PGO is therefore most useful for a *fixed* deployment: a known, stable set of
models you serve in production. Profile that set once and ship one optimized
binary.

## Requirements

The merge step needs `llvm-profdata`, shipped by the rustup component:

```sh
rustup component add llvm-tools-preview
```

It installs under the active toolchain, e.g.
`$(rustc --print sysroot)/lib/rustlib/<host>/bin/llvm-profdata`.

## Recipe

The flow is: build an instrumented binary, run it on representative models to
collect raw profiles, merge them, then rebuild using the merged profile. Keep
the three builds in separate target directories so they do not clobber each
other.

**1. Instrumented build** — writes a `.profraw` per run into `$PROFDIR`:

```sh
PROFDIR=$(pwd)/pgo-profraw
CARGO_TARGET_DIR=target-pgo-inst \
  RUSTFLAGS="-Cprofile-generate=$PROFDIR" \
  cargo build --release -p tract-cli
```

**2. Collect** — run the instrumented `tract` on each deployment model. Use
the same command shape you serve with; `bench` exercises the eval loop
repeatedly, which is what you want to profile. Profile a *representative
basket* of your models, not just one — the shared engine glue dominates the
result, so a diverse basket transfers well and per-model specialization buys
little.

```sh
BIN=target-pgo-inst/release/tract
for m in mobilenetv2-7 my_other_model; do
  LLVM_PROFILE_FILE="$PROFDIR/$m-%p.profraw" \
    "$BIN" "$m.onnx" -O bench --allow-random-input
done
```

**3. Merge** the raw profiles into one `.profdata`:

```sh
llvm-profdata merge -o pgo.profdata "$PROFDIR"/*.profraw
```

**4. Optimized build** — rebuild with the merged profile:

```sh
CARGO_TARGET_DIR=target-pgo-opt \
  RUSTFLAGS="-Cprofile-use=$(pwd)/pgo.profdata" \
  cargo build --release -p tract-cli
```

`target-pgo-opt/release/tract` is the production binary. The instrumented
binary and the `pgo-profraw`/`.profdata` artifacts can be discarded.

## Notes

- The profile must come from the same source tree as the optimized build. After
  changing tract, recollect.
- WebAssembly targets see little benefit: the host engine's JIT decides the
  final code layout, so the machine-code-layout part of PGO does not carry
  through. Prefer `simd128` and `wasm-opt` there.
