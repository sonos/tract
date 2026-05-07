//! WASM kernel microbenches. Run on wasm32 only.
//!
//!   RUSTFLAGS='-C target-feature=+simd128' \
//!     CARGO_TARGET_WASM32_WASIP1_RUNNER='wasmtime --env RUST_TEST_NOCAPTURE=1 --' \
//!     cargo bench --release --target wasm32-wasip1 -p tract-linalg --bench wasm
//!
//! Re-run with `+simd128,+relaxed-simd` to compare baseline mul+add against
//! the FMA emit driven by the `madd_f32x4!` macro in `linalg/src/wasm.rs`.

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    eprintln!("this bench only runs on wasm32 targets — skipping on host");
}

#[cfg(target_arch = "wasm32")]
fn main() {
    let target = if cfg!(target_feature = "relaxed-simd") {
        "+simd128,+relaxed-simd (FMA)"
    } else {
        "+simd128 only (mul+add)"
    };

    eprintln!("=== WASM 8x8 GEMM microbench ({target}) ===");
    bench_8x8::run();

    eprintln!();
    eprintln!("=== Isolated 32x1 GEMV microbench ({target}) ===");
    bench_32x1::run();
}

#[cfg(target_arch = "wasm32")]
mod bench_8x8 {
    //! Microbench: time `wasm_f32_8x8` (the GEMM kernel for N>=2) at shapes
    //! relevant to DFN3, transformer FFN, and CNN→GEMM workloads.

    use std::time::Instant;
    use tract_data::internal::*;
    use tract_linalg::mmm::{AsInputValue, FusedSpec};

    fn run_one(
        kernel: &dyn tract_linalg::mmm::MatMatMul,
        m: usize,
        k: usize,
        n: usize,
        iters: usize,
    ) -> f64 {
        let packing = &kernel.packings()[0];
        let a = Tensor::zero::<f32>(&[m, k]).unwrap();
        let pa = packing.0.prepare_one(&a, 1, 0).unwrap();
        let b = Tensor::zero::<f32>(&[k, n]).unwrap();
        let pb = packing.1.prepare_one(&b, 0, 1).unwrap();
        let mut c = Tensor::zero::<f32>(&[m, n]).unwrap();

        for _ in 0..50 {
            unsafe {
                kernel
                    .run(
                        m,
                        n,
                        &[
                            FusedSpec::AddMatMul {
                                a: AsInputValue::Borrowed(&*pa),
                                b: AsInputValue::Borrowed(&*pb),
                                packing: 0,
                            },
                            FusedSpec::Store(kernel.c_view(Some(0), Some(1)).wrap(&c.view_mut())),
                        ],
                    )
                    .unwrap();
            }
        }

        let t0 = Instant::now();
        for _ in 0..iters {
            unsafe {
                kernel
                    .run(
                        m,
                        n,
                        &[
                            FusedSpec::AddMatMul {
                                a: AsInputValue::Borrowed(&*pa),
                                b: AsInputValue::Borrowed(&*pb),
                                packing: 0,
                            },
                            FusedSpec::Store(kernel.c_view(Some(0), Some(1)).wrap(&c.view_mut())),
                        ],
                    )
                    .unwrap();
            }
        }
        let elapsed = t0.elapsed();
        elapsed.as_secs_f64() / iters as f64 * 1e9
    }

    fn pick(name: &str) -> Box<dyn tract_linalg::mmm::MatMatMul> {
        let mut ops = tract_linalg::generic();
        tract_linalg::wasm::plug(&mut ops);
        for impl_ in ops.mmm_impls() {
            if impl_.name() == name {
                return impl_.clone();
            }
        }
        panic!("kernel {name} not registered")
    }

    fn bench_shape(label: &str, m: usize, k: usize, n: usize, iters: usize) {
        let k88 = pick("wasm_f32_8x8");
        let ns = run_one(&*k88, m, k, n, iters);
        let m_tiles = m.div_ceil(8);
        let n_tiles = n.div_ceil(8);
        let total_tiles = m_tiles * n_tiles;
        let per_tile_ns = ns / total_tiles as f64;
        eprintln!(
            "{label} (m={m} k={k} n={n}, iters={iters}): {ns:.0} ns/call \
             ({total_tiles} 8x8 tiles, {per_tile_ns:.1} ns/tile)"
        );
    }

    pub fn run() {
        // DFN3 N>1 GEMM case (the primary 8x8 hit on DFN3).
        bench_shape("DFN3-style m=64 k=64 n=8", 64, 64, 8, 50_000);
        // Larger N — typical batched/transformer GEMM.
        bench_shape("m=64 k=64 n=64", 64, 64, 64, 10_000);
        bench_shape("m=128 k=128 n=8", 128, 128, 8, 20_000);
        bench_shape("m=128 k=128 n=64", 128, 128, 64, 5_000);
        bench_shape("m=256 k=256 n=8", 256, 256, 8, 5_000);
        bench_shape("m=256 k=256 n=64", 256, 256, 64, 1_000);
        // Whisper-tiny FFN-ish (large K, small N).
        bench_shape("m=384 k=1536 n=8", 384, 1536, 8, 1_000);
    }
}

#[cfg(target_arch = "wasm32")]
mod bench_32x1 {
    //! Isolated, statistics-aware microbench for `wasm_f32_32x1` to investigate
    //! the apparent regression at M=100/256 in `microbench_dispatch_gemv`. That
    //! bench loops all 4 GEMV kernels back-to-back at every shape, biasing the
    //! later-running kernel (32x1) with cache contention and thermal buildup.
    //! This module benches 32x1 alone, with min-of-N reporting across
    //! repetitions to expose variance honestly.

    use std::time::Instant;
    use tract_data::internal::*;
    use tract_linalg::mmm::{AsInputValue, FusedSpec};

    fn run_one(kernel: &dyn tract_linalg::mmm::MatMatMul, m: usize, k: usize, iters: usize) -> f64 {
        let packing = &kernel.packings()[0];
        let a = Tensor::zero::<f32>(&[m, k]).unwrap();
        let pa = packing.0.prepare_one(&a, 1, 0).unwrap();
        let b = Tensor::zero::<f32>(&[k, 1]).unwrap();
        let pb = packing.1.prepare_one(&b, 0, 1).unwrap();
        let mut c = Tensor::zero::<f32>(&[m, 1]).unwrap();

        // Generous warmup — 200 calls primes the JIT and hot caches.
        for _ in 0..200 {
            unsafe {
                kernel
                    .run(
                        m,
                        1,
                        &[
                            FusedSpec::AddMatMul {
                                a: AsInputValue::Borrowed(&*pa),
                                b: AsInputValue::Borrowed(&*pb),
                                packing: 0,
                            },
                            FusedSpec::Store(kernel.c_view(Some(0), Some(0)).wrap(&c.view_mut())),
                        ],
                    )
                    .unwrap();
            }
        }

        let t0 = Instant::now();
        for _ in 0..iters {
            unsafe {
                kernel
                    .run(
                        m,
                        1,
                        &[
                            FusedSpec::AddMatMul {
                                a: AsInputValue::Borrowed(&*pa),
                                b: AsInputValue::Borrowed(&*pb),
                                packing: 0,
                            },
                            FusedSpec::Store(kernel.c_view(Some(0), Some(0)).wrap(&c.view_mut())),
                        ],
                    )
                    .unwrap();
            }
        }
        let elapsed = t0.elapsed();
        elapsed.as_secs_f64() / iters as f64 * 1e9
    }

    fn pick(name: &str) -> Box<dyn tract_linalg::mmm::MatMatMul> {
        let mut ops = tract_linalg::generic();
        tract_linalg::wasm::plug(&mut ops);
        for impl_ in ops.mmm_impls() {
            if impl_.name() == name {
                return impl_.clone();
            }
        }
        panic!("kernel {name} not registered")
    }

    fn bench_min_of_n(label: &str, m: usize, k: usize, iters: usize, repetitions: usize) {
        let kernel = pick("wasm_f32_32x1");
        let mut samples: Vec<f64> = Vec::with_capacity(repetitions);
        for _ in 0..repetitions {
            samples.push(run_one(&*kernel, m, k, iters));
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let min = samples[0];
        let median = samples[samples.len() / 2];
        let max = samples[samples.len() - 1];
        let pct_spread = (max - min) / min * 100.0;
        eprintln!(
            "{label} (m={m} k={k}, {iters} iters × {repetitions} reps): \
             min={min:.0} median={median:.0} max={max:.0} ns/call (spread {pct_spread:.0}%)"
        );
    }

    pub fn run() {
        // Suspect shapes from microbench_dispatch_gemv (apparent regression):
        bench_min_of_n("M=100 k=256", 100, 256, 10_000, 10);
        bench_min_of_n("M=256 k=256", 256, 256, 5_000, 10);
        bench_min_of_n("M=256 k=512", 256, 512, 2_000, 10);
        // Reference shapes (showed clean speedup before):
        bench_min_of_n("M=24 k=256", 24, 256, 30_000, 10);
        bench_min_of_n("M=64 k=96", 64, 96, 20_000, 10);
    }
}
