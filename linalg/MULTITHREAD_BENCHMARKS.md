# Multithreaded MMM benchmarks

Validation data for the `multithread-mm` rayon path with this PR's
`chunked_dispatch_rayon` + `THREADING_PANEL_THRESHOLD` + `RayonGlobal`
changes.

## Setup

- **Base**: tract `main` (commit `41b7b02`), with the merged WASM kernel
  kit (PRs `#2164` + `#2173`).
- **Vanilla baseline**: same commit, MMM dispatch unchanged
  (1D `into_par_iter` over single panel axis).
- **Patched**: this PR applied — chunked 2D dispatch, threshold, RayonGlobal.
- **Both binaries built identically**: same compiler, same kernel kit, same
  `+atomics +bulk-memory +mutable-globals +simd128` target features for
  WASM.
- **Driver**: Playwright headless, real browser engines. Median of 60
  iterations after 3-iter warmup.
- **Output verification**: FNV-1a hash of result tensor. All 60 cells
  produce identical hash (`20ea4579c427f925` for DFN3,
  shape-deterministic for synthetic) — bit-equal output preserved.

## Synthetic dense matmul (the parallelism-bound case)

| Shape | Engine | Vanilla (1 thread) | Patched, 4 threads | Speedup |
|---|---|---|---|---|
| **1024×1024×1024** (transformer FFN) | Chromium | 55.4 ms | 16.6 ms | **3.34×** |
|  | WebKit | 79.8 ms | 25.6 ms | **3.12×** |
|  | Firefox | 1107 ms | 325 ms | **3.40×** |
| **512×768×768** (BERT FFN) | Chromium | 15.7 ms | 5.1 ms | **3.07×** |
|  | WebKit | 22.6 ms | 6.8 ms | **3.30×** |
|  | Firefox | 311 ms | 91 ms | **3.42×** |
| **256×256×128** | Chromium | 0.47 ms | 0.22 ms | **2.19×** |
|  | WebKit | 0.66 ms | 0.20 ms | **3.30×** |
|  | Firefox | 9.0 ms | 2.6 ms | **3.41×** |
| 64×256×64 (DFN-like small) | Chromium | 0.07 ms | 0.07 ms | 1.0× (within noise) |
|  | WebKit | 0.08 ms | 0.04 ms | 2.00× |
|  | Firefox | 1.18 ms | 0.38 ms | **3.11×** |
| 32×32×32 (tiny) | All | sub-ms | sub-ms | **1.0× (threshold gates)** |

The threshold correctly gates the smallest shape; threading kicks in once
panel count clears the gate, and scales near-linearly with thread count
on all three engines.

## Real model (DeepFilterNet 3, full streaming inference)

5-frame chunks at 48 kHz (50 ms of audio per iteration).

| Engine | Vanilla mono | Patched, 4 threads | RTF (vanilla → patched) | Speedup |
|---|---|---|---|---|
| Chromium | 3.31 ms | 3.17 ms | 0.066 → 0.063 | 1.04× (within noise) |
| WebKit | 4.32 ms | 4.00 ms | 0.086 → 0.080 | 1.08× |
| Firefox | 34.20 ms | 33.34 ms | 0.684 → 0.667 | 1.03× |

DFN3 is Amdahl-bound: only ~25% of runtime is in MMMs that clear the
threshold; the rest is FFT, complex multiplication, and small RNN-internal
matmuls that the threshold deliberately keeps single-threaded. The
threshold's role here is to **prevent regression**, not deliver speedup.
This is correct behavior — DFN3-class workloads should not pay rayon
overhead on tiny ops.

## Native (macOS aarch64, generic kernels)

Spot-check on the rayon path before/after. Tract's existing rayon path
already worked well on native; the change is mostly a refactor.

| Shape | Vanilla 1D | Patched 2D | Change |
|---|---|---|---|
| 256×256, 4 threads | 2.13 ms | 2.12 ms | net-neutral |
| 512×512, 4 threads | 9.93 ms | 9.80 ms | +1% |
| 64×256, 4 threads | 615 µs | 625 µs | −2% |

Within noise on common shapes. The 2D dispatch shows a latent benefit on
shapes 1D parallelism handles poorly (e.g. m=8 n=2048, where 1D over m
can only feed 2 threads); not yet measured directly on native but the
dispatch math is the same on both targets.

## Determinism

Across all measured cells (synthetic + DFN3, 3 engines, 1/2/3/4 threads):

- **WASM**: 60 cells, all produce identical hash per `(shape, mode)` pair.
- **Native**: existing tract proptests (3524 lib tests) pass with this
  PR's `multithread-mm` enabled.

## Tuning the threshold

The default `THREADING_PANEL_THRESHOLD` is `64` panels (m_panels ×
n_panels). Adjust at runtime via:

```rust
use tract_linalg::multithread::set_threading_panel_threshold;

set_threading_panel_threshold(0);    // thread every size, no gate
set_threading_panel_threshold(256);  // gate harder — transformer-only
set_threading_panel_threshold(64);   // default
```

Useful when profiling or specialising the build for a known workload class:

| Workload class | Suggested threshold |
|---|---|
| Streaming RNN / mobile vision (many small MMMs) | 64 (default) or higher |
| Mid-size dense (BERT-class) | 32–64 |
| Large dense only (transformer FFN, LLM) | 16 or lower |

The constant lives in `linalg/src/multithread.rs`; readers go through
`current_threading_panel_threshold()` (`AtomicUsize::Relaxed`, no lock on
the dispatch hot path).

## Reproduction

The harness uses [Vonage's libDF fork](https://github.com/czoli1976/DeepFilterNet)
(branch `dfn3-wasm-opt-tract-022-kernel-kit`) migrated to tract main, with
a `wasm-bindgen-rayon`-based threading bootstrap. Build:

```bash
RUSTFLAGS="-C target-feature=+atomics,+bulk-memory,+mutable-globals,+simd128" \
wasm-pack build --target web --release \
    --no-default-features --features wasm-mt -- \
    -Z build-std=std,panic_abort
```

JS-side:

```javascript
import init, { initThreadPool, df_set_thread_count } from './pkg/df.js';
await init();
await initThreadPool(navigator.hardwareConcurrency);  // wasm-bindgen-rayon
df_set_thread_count(4);  // sets Executor::RayonGlobal in tract-linalg
```

Without `Executor::RayonGlobal` (this PR), `df_set_thread_count` would
need to construct an `Arc<rayon::ThreadPool>` — which fails on
`wasm32-unknown-unknown` because `rayon::ThreadPoolBuilder::new().build()`
internally calls `std::thread::spawn` (unsupported there). That's the
crux of why this enabling change is needed in tract-linalg itself: any
browser threading via wasm-bindgen-rayon would otherwise silently fall
back to single-threaded.
