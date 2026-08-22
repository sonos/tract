#[cfg(test)]
mod dispatch_trace {
    fn trace_one(label: &str, m: Option<usize>, k: Option<usize>, n: Option<usize>) {
        let mut ops = crate::generic();
        crate::wasm::plug(&mut ops);
        let mmm = (ops.mmm_policy())(tract_data::prelude::DatumType::F32, m, k, n).unwrap();
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
    /// otherwise it falls through to the candidate list, where
    /// `retain_best_quality` drops every `TargetOptimized` kernel, and for
    /// N>1 then picks `max(nr*mr)`
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
                (ops.mmm_policy())(tract_data::prelude::DatumType::F32, Some(m), Some(k), Some(n))
                    .unwrap();
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

/// `AddRowColProducts` and `AddMatMul` with `k = 1` both compute
/// `c[i][j] += a[i] * b[j]`, so a kernel has to use the same multiply-add form
/// for both arms. Under `+relaxed-simd` the fused form keeps the full product
/// before adding while the separate form rounds it first, so a kernel that
/// fuses one arm and not the other returns two different answers for the same
/// arithmetic.
///
/// The operands make that difference observable: with `a = b = 1 + 2^-12` the
/// product `1 + 2^-11 + 2^-24` needs 25 significand bits and rounds to
/// `1 + 2^-11`, so against `c = -1` the fused form yields `2^-11 + 2^-24` and
/// the separate form `2^-11`. Without `+relaxed-simd` both arms are `mul` then
/// `add` and the two agree trivially.
#[cfg(test)]
fn check_madd_pairing<K: crate::mmm::MatMatMulKer<Acc = f32>>(ker: &K) {
    use crate::mmm::{FusedKerSpec, OutputStoreKer};

    if !ker.is_supported_here() {
        return;
    }
    let (mr, nr) = (ker.mr(), ker.nr());
    let v = 1f32 + 2f32.powi(-12);

    let (pack_a, pack_b) = &ker.packings()[0];
    let k = pack_a.k_alignment().max(pack_b.k_alignment());
    let mut a_data = vec![0f32; mr * k];
    let mut b_data = vec![0f32; k * nr];
    for i in 0..mr {
        a_data[i * k] = v;
    }
    b_data[..nr].copy_from_slice(&vec![v; nr]);
    let a = Tensor::from_shape(&[mr, k], &a_data).unwrap();
    let b = Tensor::from_shape(&[k, nr], &b_data).unwrap();
    let pa = pack_a.prepare_one(&a, 1, 0).unwrap();
    let pb = pack_b.prepare_one(&b, 0, 1).unwrap();

    let rows = vec![v; mr];
    let cols = vec![v; nr];

    let run = |op: FusedKerSpec<f32>| -> Vec<f32> {
        let out = vec![0f32; mr * nr];
        let item = std::mem::size_of::<f32>();
        let store = OutputStoreKer {
            ptr: out.as_ptr() as *mut u8,
            row_byte_stride: (item * nr) as isize,
            col_byte_stride: item as isize,
            item_size: item,
        };
        let ops = [
            FusedKerSpec::Clear,
            FusedKerSpec::ScalarAdd(-1.0),
            op,
            FusedKerSpec::Store(store),
            FusedKerSpec::Done,
        ];
        assert_eq!(ker.kernel(&ops), 0);
        out
    };

    let from_row_col = run(FusedKerSpec::AddRowColProducts(rows.as_ptr(), cols.as_ptr()));
    let from_mat_mul = run(FusedKerSpec::AddMatMul {
        k,
        pa: pa.panel_bytes(0, None).unwrap(),
        pb: pb.panel_bytes(0, None).unwrap(),
        packing: 0,
    });

    for (i, (rc, mm)) in from_row_col.iter().zip(from_mat_mul.iter()).enumerate() {
        assert_eq!(
            rc.to_bits(),
            mm.to_bits(),
            "{}: cell {i} is {rc:e} from AddRowColProducts but {mm:e} from AddMatMul — \
             the two arms disagree on whether the multiply-add is fused",
            ker.name()
        );
    }
}

#[test]
fn add_row_col_products_and_add_mat_mul_agree_on_fusion() {
    check_madd_pairing(&*crate::wasm::wasm_f32_4x4);
    check_madd_pairing(&*crate::wasm::wasm_f32_4x1);
    check_madd_pairing(&*crate::wasm::wasm_f32_8x1);
    check_madd_pairing(&*crate::wasm::wasm_f32_16x1);
    check_madd_pairing(&*crate::wasm::wasm_f32_32x1);
    check_madd_pairing(&*crate::wasm::wasm_f32_8x8);
}

/// `wasm_f32_4x4` is registered at `TargetOptimized` while every other kernel
/// is `ManuallyOptimized`, so `strategize`'s `retain()` drops it before
/// selection and neither `mmm_f32` nor `mmv_f32` ever names it. Promoting it
/// without also giving it a dispatch band would silently put a 4-wide tile in
/// front of shapes the 8x8 and the GEMV kernels currently own.
#[test]
fn dispatch_never_returns_wasm_f32_4x4() {
    let mut ops = crate::generic();
    crate::wasm::plug(&mut ops);
    for m in [1usize, 3, 4, 5, 8, 9, 16, 17, 32, 64, 256, 1024] {
        for n in [1usize, 2, 4, 8, 10, 64, 256] {
            for k in [1usize, 64, 576] {
                let mmm = ops
                    .mmm(tract_data::prelude::DatumType::F32, Some(m), Some(k), Some(n))
                    .unwrap();
                assert_ne!(
                    mmm.name(),
                    "wasm_f32_4x4",
                    "m={m} k={k} n={n} dispatched to wasm_f32_4x4, which is registered \
                     TargetOptimized and has no dispatch band"
                );
            }
        }
    }
}
