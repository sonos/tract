# tract-linalg on `wasm32` — relaxed-simd FMA

The WASM MMM kernels (`wasm_f32_4x4`, `4x1`, `8x1`, `16x1`, `32x1`, `8x8`)
and the WASM sigmoid/tanh activations all flip between two emit modes at
compile time, gated on `cfg(target_feature = "relaxed-simd")`:

- **Without** `+relaxed-simd`: pure `f32x4_add(_, f32x4_mul(_, _))` (mul+add).
  Runs on any WASM runtime that supports `simd128`.
- **With** `+relaxed-simd`: `f32x4_relaxed_madd(_, _, _)`. Fused, single-rounded
  multiply-add on hosts whose CPU has hardware FMA (all ARM64, x86_64 + FMA3).
  Universal browser/runtime support since 2023 (Chrome 114+, Firefox 120+,
  Safari 17+, wasmtime 16+).

The speedup of the relaxed path over the baseline is typically **1.40–1.55× at
the kernel level** and **1.08–1.46× end-to-end** across vision CNNs,
transformer attention and RNN audio models. Bit-pattern drift versus the
mul+add path is bounded at one ulp (FMA single-rounding); within
`Approximation::Close` (1e-4).

## Build flags

```sh
# Baseline (any wasm32 runtime supporting simd128)
RUSTFLAGS='-C target-feature=+simd128' \
  cargo build --release --target wasm32-wasip1 -p tract-linalg

# Relaxed (requires host support for relaxed-simd; ~1.40× faster on FMA-capable hosts)
RUSTFLAGS='-C target-feature=+simd128,+relaxed-simd' \
  cargo build --release --target wasm32-wasip1 -p tract-linalg
```

Same on `wasm32-unknown-unknown` if shipping for the browser.

## Why two binaries (and not in-process runtime dispatch)

WASM validates the entire module at instantiation, before any code runs.
A binary containing `f32x4.relaxed_madd` fails to instantiate on hosts without
relaxed-simd — `LinkError` / `CompileError`, not a runtime trap. So the
x86/ARM pattern (one binary, both paths in source, runtime CPU detection picks
at execution time) cannot be replicated in-binary on WASM: the FMA opcodes are
either present (and host support is required) or absent.

Runtime dispatch happens one layer up — at the host runtime / consumer layer
— by selecting the correct binary at module-load time.

## Consumer-side dispatch

### Browser / `WebAssembly.validate`

```js
async function loadTract(baseUrl) {
    const candidate = await fetch(`${baseUrl}/tract-relaxed.wasm`);
    const bytes = await candidate.arrayBuffer();

    const wantRelaxed = WebAssembly.validate(bytes, {
        builtins: ['relaxed_simd'],
    });

    const url = wantRelaxed
        ? `${baseUrl}/tract-relaxed.wasm`
        : `${baseUrl}/tract.wasm`;

    const final = await fetch(url);
    return WebAssembly.instantiateStreaming(final);
}
```

Fallback for hosts without the `WebAssembly.validate(bytes, { ... })`
options-arg: try-instantiate the relaxed binary, catch `LinkError` /
`CompileError`, retry with the baseline.

### `wasmtime` (server / native)

```rust
use wasmtime::{Config, Engine};

let mut config = Config::new();
config.wasm_relaxed_simd(true);     // gate on host-CPU detection if needed
let engine = Engine::new(&config)?;

let bytes = std::fs::read(if relaxed_supported {
    "tract-relaxed.wasm"
} else {
    "tract.wasm"
})?;
let module = wasmtime::Module::new(&engine, &bytes)?;
```

`wasmtime::Engine`'s `wasm_relaxed_simd` configures the runtime; a separate
`wasmtime::Module::validate()` call against the engine is the equivalent of
the browser's `WebAssembly.validate` for picking which binary to load.

## Quality

The two binaries are **not bit-identical**. FMA's single-rounding produces
≤1 ulp drift from explicit mul+add. Verified end-to-end on Inception v3 and
DFN3 sub-models:

| model        | output shape       | baseline L2  | relaxed L2   |
|--------------|--------------------|-------------:|-------------:|
| Inception v3 | [1, 1001]          | 6.477089e-2  | 6.477089e-2  |
| DFN3 df_dec  | [1, 100, 96, 10]   | 1.080686e-2  | 1.080686e-2  |

L2 norms are bit-identical to 7 sig figs; per-element values diverge in the
7th–8th decimal place. Within tract's `Approximation::Close` (1e-4).
