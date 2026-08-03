#[cfg(test)]
mod dispatch_trace {
    fn trace_one(label: &str, m: Option<usize>, k: Option<usize>, n: Option<usize>) {
        let mut ops = crate::generic();
        crate::wasm::plug(&mut ops);
        let mmm = ops.mmm(tract_data::prelude::DatumType::F32, m, k, n).unwrap();
        eprintln!(
            "DFN3 {} (m={:?} k={:?} n={:?}) => {}  [mr={}, nr={}]",
            label,
            m,
            k,
            n,
            mmm.name(),
            mmm.mr(),
            mmm.nr()
        );
    }

    #[test]
    fn dfn3_shapes() {
        // DFN3 N=1 GEMV ops (the dominant matrix-vector cases)
        trace_one("lsnr_fc-style m=1 k=512", Some(1), Some(512), Some(1));
        trace_one("small m=16 k=96", Some(16), Some(96), Some(1));
        trace_one("medium m=32 k=256", Some(32), Some(256), Some(1));
        trace_one("GRU m=256 k=256", Some(256), Some(256), Some(1));
        trace_one("post-rnn m=256 k=512", Some(256), Some(512), Some(1));
        trace_one("frame-encoder m=64 k=96", Some(64), Some(96), Some(1));
        // N>1 sanity: should hit 8x8
        trace_one("MM m=64 k=64 n=8", Some(64), Some(64), Some(8));
    }

    /// Exercise every M-band edge of mmv_f32 to lock in the dispatch.
    /// Lower edge of each band = perfect-tile size; upper edge = last
    /// M before crossover to the next kernel.
    #[test]
    fn band_edges() {
        // 4x1 band: M ∈ 0..=4
        trace_one("band 4x1 lo m=1", Some(1), Some(64), Some(1));
        trace_one("band 4x1 hi m=4", Some(4), Some(64), Some(1));
        // 8x1 band: M ∈ 5..=8
        trace_one("band 8x1 lo m=5", Some(5), Some(64), Some(1));
        trace_one("band 8x1 hi m=8", Some(8), Some(64), Some(1));
        // 16x1 band: M ∈ 9..=16
        trace_one("band 16x1 lo m=9", Some(9), Some(64), Some(1));
        trace_one("band 16x1 hi m=16", Some(16), Some(64), Some(1));
        // 32x1 band: M ≥ 17
        trace_one("band 32x1 lo m=17", Some(17), Some(64), Some(1));
        trace_one("band 32x1 hi m=512", Some(512), Some(64), Some(1));
    }

    /// Regression guard for the GEMM/GEMV dispatch.
    ///
    /// `kernel_selection::strategize` honours the `mmm_f32` / `mmv_f32`
    /// callback only when the returned kernel is `ManuallyOptimized`;
    /// otherwise it falls through to `list_impls`, whose `retain()` drops
    /// every `TargetOptimized` kernel, and for N>1 then picks `max(nr*mr)`
    /// over the surviving `ManuallyOptimized` GEMV kernels — i.e.
    /// `wasm_f32_32x1`, a matrix×vector kernel, for every GEMM. So every
    /// kernel reachable through the dispatch callbacks must be
    /// `ManuallyOptimized`.
    #[test]
    fn dispatch_kernels_are_manually_optimized() {
        use crate::mmm::ImplementationQuality::ManuallyOptimized;
        let mut ops = crate::generic();
        crate::wasm::plug(&mut ops);
        for (label, m, k, n) in [
            ("GEMM m=64 k=64 n=8", 64, 64, 8),
            ("GEMM m=256 k=256 n=256", 256, 256, 256),
            ("GEMM m=1024 k=576 n=10", 1024, 576, 10),
            ("GEMV m=1 k=512 n=1", 1, 512, 1),
            ("GEMV m=256 k=256 n=1", 256, 256, 1),
        ] {
            let mmm =
                ops.mmm(tract_data::prelude::DatumType::F32, Some(m), Some(k), Some(n)).unwrap();
            assert_eq!(
                mmm.quality(),
                ManuallyOptimized,
                "{label}: dispatch returned {} tagged {:?} — strategize would \
                 discard it and reroute onto a GEMV kernel",
                mmm.name(),
                mmm.quality(),
            );
        }
    }
}
use crate::mmm::{AsInputValue, FusedSpec};
use tract_data::internal::*;

fn pick(name: &str) -> Box<dyn crate::mmm::MatMatMul> {
    let mut ops = crate::generic();
    crate::wasm::plug(&mut ops);
    for impl_ in ops.mmm_impls() {
        if impl_.name() == name {
            return impl_.clone();
        }
    }
    panic!("kernel {name} not registered")
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
