/// Wasm SIMD implementation of `MatMatMulKer<f32>`
///
/// To run test, you need to install `wasmtime`
/// and export the following environment variables:
/// ```
/// > export RUSTFLAGS='-C target-feature=+simd128'
/// > export CARGO_TARGET_WASM32_WASI_RUNNER=wasmtime
/// > cargo test --target=wasm32-wasi
/// ```
use crate::Ops;

#[cfg(target_feature = "relaxed-simd")]
use crate::frame::element_wise::ElementWiseKer;

#[macro_use]
mod madd;

#[cfg(target_feature = "relaxed-simd")]
mod act;
#[cfg(test)]
mod dispatch_tests;
mod mmm_f32_gemm;
mod mmm_f32_gemv;
mod mmm_i32;

#[cfg(target_feature = "relaxed-simd")]
pub use act::*;
pub use mmm_f32_gemm::*;
pub use mmm_f32_gemv::*;
pub use mmm_i32::*;

pub fn plug(ops: &mut Ops) {
    ops.mmm_impls.push(wasm_f32_4x4.mmm());
    ops.mmm_impls.push(wasm_f32_4x1.mmm());
    ops.mmm_impls.push(wasm_f32_8x1.mmm());
    ops.mmm_impls.push(wasm_f32_16x1.mmm());
    ops.mmm_impls.push(wasm_f32_32x1.mmm());
    ops.mmm_impls.push(wasm_f32_8x8.mmm());
    // int8 -> i32 matmul: SIMD kernel (was generic scalar). ManuallyOptimized so
    // strategize's retain() keeps it over generic_i32_4x4 for i8 packing.
    ops.mmm_impls.push(wasm_i32_4x4.mmm());
    ops.qmmm_i32 = Box::new(|_, _, _| wasm_i32_4x4.mmm());
    // Selection paths. Both rely on kernel_selection::strategize honouring
    // the mmm_f32 / mmv_f32 callback, which it only does when the callback's
    // kernel is tagged ManuallyOptimized. Otherwise strategize falls through
    // to list_impls, whose retain() keeps only the top ImplementationQuality
    // and drops every TargetOptimized kernel.
    //   - N>1 (GEMM): mmm_f32 returns 8x8, so 8x8 MUST be ManuallyOptimized.
    //     If it were TargetOptimized it would be dropped by retain(), and the
    //     N>1 branch's max(nr*mr) over the surviving (ManuallyOptimized) GEMV
    //     kernels would pick wasm_f32_32x1 — a matrix×vector kernel — for
    //     every GEMM.
    //   - N=1 (GEMV): mmv_f32 routes by M-band to the kernel whose MR fits.
    //     The four GEMV kernels are ManuallyOptimized for the same reason —
    //     without the tag strategize discards the callback and picks
    //     max(mr)=32x1 for every M, leaving up to ~37% on the table for
    //     small-M GEMV.
    ops.mmm_f32 = Box::new(|_m, _k, _n| wasm_f32_8x8.mmm());
    // Bands derived from microbench_dispatch_gemv. At each band edge, using
    // the next-larger kernel beats halving outer iterations of the smaller
    // one (1 outer with ILP-absorbed padding > 2 outer with kernel preamble
    // doubled). M=4/8/16 are exact tile fits at the lower edges; M=17/9/5
    // are the first values where the next-larger kernel wins.
    ops.mmv_f32 = Box::new(|m, _k| match m.unwrap_or(0) {
        0..=4 => wasm_f32_4x1.mmm(),
        5..=8 => wasm_f32_8x1.mmm(),
        9..=16 => wasm_f32_16x1.mmm(),
        _ => wasm_f32_32x1.mmm(),
    });
    // Relaxed-SIMD activation kernels (FMA path). Only installed when the
    // build has `+relaxed-simd`; otherwise the slots stay at the generic
    // scalar polynomial.
    #[cfg(target_feature = "relaxed-simd")]
    {
        ops.sigmoid_f32 = Box::new(|| WasmSigmoid4Relaxed::ew());
        ops.tanh_f32 = Box::new(|| WasmTanh4Relaxed::ew());
    }
}

#[cfg(test)]
mod microbench_32x1 {
    //! Quick microbench: time per-call cost for the kernel kit's GEMV path
    //! on DFN3-shaped inputs. Compares 16x1 vs 32x1 head-to-head by
    //! dispatching the named kernel directly.
    //!
    //! Run with:
    //!   RUSTFLAGS='-C target-feature=+simd128' \
    //!     CARGO_TARGET_WASM32_WASIP1_RUNNER='wasmtime --env RUST_TEST_NOCAPTURE=1 --' \
    //!     cargo test --release --target=wasm32-wasip1 -p tract-linalg \
    //!     wasm::microbench_32x1::microbench -- --nocapture --ignored

    use crate::mmm::{AsInputValue, FusedSpec};
    use std::time::Instant;
    use tract_data::internal::*;
    use tract_data::prelude::*;

    fn run_one(kernel: &dyn crate::mmm::MatMatMul, m: usize, k: usize, iters: usize) -> f64 {
        // Pack A (m,k) and B (k,1)
        let packing = &kernel.packings()[0];
        let a = Tensor::zero::<f32>(&[m, k]).unwrap();
        let pa = packing.0.prepare_one(&a, 1, 0).unwrap();
        let b = Tensor::zero::<f32>(&[k, 1]).unwrap();
        let pb = packing.1.prepare_one(&b, 0, 1).unwrap();
        let mut c = Tensor::zero::<f32>(&[m, 1]).unwrap();

        // Warmup
        for _ in 0..50 {
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

        // Timed
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
        elapsed.as_secs_f64() / iters as f64 * 1e9 // ns/call
    }

    fn pick(name: &str) -> Box<dyn crate::mmm::MatMatMul> {
        let mut ops = crate::generic();
        super::plug(&mut ops);
        for impl_ in ops.mmm_impls() {
            if impl_.name() == name {
                return impl_.clone();
            }
        }
        panic!("kernel {name} not registered")
    }

    fn bench_shape(label: &str, m: usize, k: usize, iters: usize) {
        let k16 = pick("wasm_f32_16x1");
        let k32 = pick("wasm_f32_32x1");
        let ns16 = run_one(&*k16, m, k, iters);
        let ns32 = run_one(&*k32, m, k, iters);
        let calls16 = m.div_ceil(16);
        let calls32 = m.div_ceil(32);
        let delta = (ns32 - ns16) / ns16 * 100.0;
        eprintln!(
            "{label} (m={m}, k={k}, iters={iters}): 16x1={ns16:.1} ns/call ({calls16} kernel calls); 32x1={ns32:.1} ns/call ({calls32} kernel calls); Δ={delta:+.2}% ; per-frame call ns: 16x1={n16:.1} 32x1={n32:.1} pf-Δ={dpf:+.2}%",
            n16 = ns16 * calls16 as f64,
            n32 = ns32 * calls32 as f64,
            dpf = (ns32 * calls32 as f64 - ns16 * calls16 as f64) / (ns16 * calls16 as f64) * 100.0,
        );
    }

    #[test]
    #[ignore]
    fn microbench() {
        eprintln!("=== DFN3 GEMV microbench: 16x1 vs 32x1 ===");
        // DFN3 GRU gates (highest call count)
        bench_shape("GRU m=256 k=256", 256, 256, 5_000);
        // post-RNN
        bench_shape("post-rnn m=256 k=512", 256, 512, 3_000);
        // frame encoder
        bench_shape("frame-encoder m=64 k=96", 64, 96, 20_000);
        // perfect tile
        bench_shape("perfect-tile m=32 k=256", 32, 256, 20_000);
    }

    /// Numerical-equivalence sanity check between 16x1 and 32x1 kernels on a
    /// real-shape matmul with non-trivial inputs.
    ///
    /// Under `+simd128` (no relaxed-simd): both kernels emit
    /// `f32x4_add(f32x4_mul(...))` via `madd_f32x4!`, so the K-loop order is
    /// identical and outputs are bit-identical.
    ///
    /// Under `+simd128,+relaxed-simd`: 32x1 uses `f32x4.relaxed_madd` (fused
    /// FMA) via `madd_f32x4!`, while 16x1 uses separate `mul+add` via
    /// `madd_f32x4_nofma!` to avoid the destructive-accumulator recurrence
    /// that throttles ≤4-accumulator kernels (see header comment on
    /// `madd_f32x4_nofma`). Outputs drift by ≤1 ulp per K-step from the
    /// rounding difference between fused and separate ops. We accept that
    /// drift with a generous relative tolerance.
    #[test]
    fn numerical_consistency_16x1_vs_32x1() {
        let m = 256usize;
        let k = 256usize;
        let mut a_data = vec![0f32; m * k];
        for (i, x) in a_data.iter_mut().enumerate() {
            *x = ((i % 13) as f32 - 6.0) * 0.1 + ((i / 17) % 11) as f32 * 0.07;
        }
        let mut b_data = vec![0f32; k];
        for (i, x) in b_data.iter_mut().enumerate() {
            *x = (i as f32).sin() * 0.5;
        }
        let a = Tensor::from_shape(&[m, k], &a_data).unwrap();
        let b = Tensor::from_shape(&[k, 1], &b_data).unwrap();

        let run = |name: &str| -> Vec<f32> {
            let kernel = pick(name);
            let packing = &kernel.packings()[0];
            let pa = packing.0.prepare_one(&a, 1, 0).unwrap();
            let pb = packing.1.prepare_one(&b, 0, 1).unwrap();
            let mut c = Tensor::zero::<f32>(&[m, 1]).unwrap();
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
            c.try_as_plain().unwrap().as_slice::<f32>().unwrap().to_vec()
        };

        let c16 = run("wasm_f32_16x1");
        let c32 = run("wasm_f32_32x1");

        #[cfg(not(target_feature = "relaxed-simd"))]
        {
            for (i, (x16, x32)) in c16.iter().zip(c32.iter()).enumerate() {
                assert!(
                    x16.to_bits() == x32.to_bits(),
                    "row {i}: 16x1={x16} (bits 0x{:x}) != 32x1={x32} (bits 0x{:x})",
                    x16.to_bits(),
                    x32.to_bits()
                );
            }
            eprintln!("bit-identity OK over m={m} k={k} ({} rows)", m);
        }

        #[cfg(target_feature = "relaxed-simd")]
        {
            // K=256 accumulator drift on fp32 between FMA and separate mul+add
            // can grow up to roughly K × 0.5 ulp ≈ 128 ulp in the accumulator.
            // For small-magnitude outputs that translates to ~1e-4 relative.
            // We use 1e-4 as the tolerance — tight enough to catch real bugs
            // (typically 1e-2+ drift) but generous for legitimate FMA drift.
            let mut max_abs = 0.0f32;
            let mut max_rel = 0.0f32;
            for (i, (x16, x32)) in c16.iter().zip(c32.iter()).enumerate() {
                let abs = (x16 - x32).abs();
                let scale = x16.abs().max(x32.abs()).max(1.0e-9);
                let rel = abs / scale;
                assert!(
                    rel < 1.0e-4,
                    "row {i}: relative drift {rel:e} too large; 16x1={x16} 32x1={x32}"
                );
                if abs > max_abs {
                    max_abs = abs;
                }
                if rel > max_rel {
                    max_rel = rel;
                }
            }
            eprintln!(
                "relaxed-simd consistency OK over m={m} k={k}: max abs={max_abs:.3e}, max rel={max_rel:.3e}"
            );
        }
    }
}

#[cfg(test)]
mod microbench_dispatch_gemv {
    //! Microbench: 4x1 vs 8x1 vs 16x1 vs 32x1 GEMV kernels across the M
    //! range. Drives the dispatch-fix decision — the M-band callback in
    //! plug() routes small-M to smaller kernels, but only takes effect
    //! once the kernels are tagged ManuallyOptimized (otherwise
    //! kernel_selection::strategize bypasses the callback and always
    //! picks max(mr) = 32x1).
    //!
    //! Run with:
    //!   RUSTFLAGS='-C target-feature=+simd128' \
    //!     CARGO_TARGET_WASM32_WASIP1_RUNNER='wasmtime --env RUST_TEST_NOCAPTURE=1 --' \
    //!     cargo test --release --target=wasm32-wasip1 -p tract-linalg \
    //!     wasm::microbench_dispatch_gemv::microbench -- --nocapture --ignored

    use crate::mmm::{AsInputValue, FusedSpec};
    use std::time::Instant;
    use tract_data::internal::*;
    use tract_data::prelude::*;

    fn run_one(kernel: &dyn crate::mmm::MatMatMul, m: usize, k: usize, iters: usize) -> f64 {
        let packing = &kernel.packings()[0];
        let a = Tensor::zero::<f32>(&[m, k]).unwrap();
        let pa = packing.0.prepare_one(&a, 1, 0).unwrap();
        let b = Tensor::zero::<f32>(&[k, 1]).unwrap();
        let pb = packing.1.prepare_one(&b, 0, 1).unwrap();
        let mut c = Tensor::zero::<f32>(&[m, 1]).unwrap();

        for _ in 0..50 {
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

    fn pick(name: &str) -> Box<dyn crate::mmm::MatMatMul> {
        let mut ops = crate::generic();
        super::plug(&mut ops);
        for impl_ in ops.mmm_impls() {
            if impl_.name() == name {
                return impl_.clone();
            }
        }
        panic!("kernel {name} not registered")
    }

    fn bench_shape(label: &str, m: usize, k: usize, iters: usize) {
        let k4 = pick("wasm_f32_4x1");
        let k8 = pick("wasm_f32_8x1");
        let k16 = pick("wasm_f32_16x1");
        let k32 = pick("wasm_f32_32x1");
        let n4 = run_one(&*k4, m, k, iters);
        let n8 = run_one(&*k8, m, k, iters);
        let n16 = run_one(&*k16, m, k, iters);
        let n32 = run_one(&*k32, m, k, iters);
        let entries = [("4x1", n4), ("8x1", n8), ("16x1", n16), ("32x1", n32)];
        let winner = entries.iter().min_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
        let delta_vs_32 = (winner.1 - n32) / n32 * 100.0;
        eprintln!(
            "{label} (m={m} k={k}): 4x1={n4:.0} 8x1={n8:.0} 16x1={n16:.0} 32x1={n32:.0} ns; \
             winner={} ({:.0} ns, Δ vs 32x1: {delta_vs_32:+.1}%)",
            winner.0, winner.1
        );
    }

    #[test]
    #[ignore]
    fn microbench() {
        eprintln!("=== WASM GEMV dispatch microbench: 4x1 vs 8x1 vs 16x1 vs 32x1 ===");
        // M ≤ 16 — small-M region; the M-band callback's choices win clearly.
        bench_shape("M=1   k=512", 1, 512, 50_000);
        bench_shape("M=8   k=64 ", 8, 64, 50_000);
        bench_shape("M=8   k=512", 8, 512, 20_000);
        bench_shape("M=12  k=256", 12, 256, 50_000);
        bench_shape("M=16  k=96 ", 16, 96, 50_000);
        bench_shape("M=16  k=256", 16, 256, 30_000);
        // M ≥ 17 — 32x1 wins (16x1 needs 2 outer iters, 32x1 single iter
        // with ILP absorbing the row padding).
        bench_shape("M=24  k=256", 24, 256, 30_000);
        bench_shape("M=32  k=256", 32, 256, 20_000);
        bench_shape("M=64  k=96 ", 64, 96, 20_000);
        bench_shape("M=100 k=256", 100, 256, 10_000);
        bench_shape("M=256 k=256", 256, 256, 5_000);
    }
}
