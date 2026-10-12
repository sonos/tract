use crate::DatumType;
use crate::isa::IsaSet;
use crate::mmm::*;
use crate::mmm_tiers::MmmTier;

// CAN_FUSE: everything except LeakyRelu / QScale / RoundingShiftRight /
// ShiftLeft. LoadTile, AddUnicast, AddRowColProducts, per-row/col/scalar
// arithmetic, Clear, Store, AddMatMul are all in. (Matches AMX
// `apple_amx.rs` CAN_FUSE, minus the i32-only quantization ops.)
const CAN_FUSE: fn(&FusedSpec) -> bool = |f| {
    !matches!(
        f,
        FusedSpec::LeakyRelu(_)
            | FusedSpec::QScale(_, _, _)
            | FusedSpec::RoundingShiftRight(_, _)
            | FusedSpec::ShiftLeft(_)
    )
};

// The SMOPA i32 kernel implements the quant fuse ops (QScale / RoundingShiftRight
// / ShiftLeft) bit-exactly; only LeakyRelu is unsupported (kernel returns 1).
const CAN_FUSE_I32: fn(&FusedSpec) -> bool = |f| !matches!(f, FusedSpec::LeakyRelu(_));

MMMExternKernel!(aarch64; sme_qmmm_i32_32x32<i32>(32,32)@(128,128) isa(Aarch64Sme2) can_fuse(CAN_FUSE_I32)
    packing[1] = i8i8 => |k| k.with_packing(crate::pack::PackedI8K4::new(32), crate::pack::PackedI8K4::new(32));
    store(i8));

// Streaming vector length in bytes, read via `RDSVL x0, #1` (encoding
// 0x04bf5820). RDSVL is legal in non-streaming mode, but is UNDEFINED
// unless FEAT_SME is implemented — callers MUST confirm FEAT_SME first
// (sysctl on macOS, HWCAP2 on Linux) or this SIGILLs.
#[cfg(any(target_os = "macos", target_os = "linux"))]
unsafe fn streaming_vector_bytes() -> u64 {
    let svl: u64;
    unsafe {
        std::arch::asm!(
            ".inst 0x04bf5820", // rdsvl x0, #1
            out("x0") svl,
            options(nomem, nostack, preserves_flags),
        );
    }
    svl
}

// Our SME kernels hardcode a 512-bit streaming vector length (16 f32 lanes
// per ZA.S slice — the 32x32 and 64x1 tile geometries depend on it). A host
// that advertises FEAT_SME with a different SVL would run the kernels with
// mismatched geometry and produce silently-wrong results. The prime offender
// is qemu-aarch64 user-mode emulation, which sets HWCAP2_SME / HWCAP2_SME2
// but uses a non-512 SVL — that is exactly what makes the cross-compiled
// aarch64 CI jobs (run under QEMU) fail. Reject any non-512 SVL here so we
// fall back to the portable path. MUST only be called once FEAT_SME is known
// present.
#[cfg(any(target_os = "macos", target_os = "linux"))]
fn sme_geometry_supported() -> bool {
    // SVL = 512 bits = 64 bytes.
    unsafe { streaming_vector_bytes() == 64 }
}

MMMExternKernel!(aarch64;
    sme_mmm_f32_32x32<f32>(32, 32)@(128, 128)
    isa(Aarch64Sme)
    can_fuse(CAN_FUSE)
    row_major_store(true)
);

MMMExternKernel!(aarch64;
    sme_mmv_f32_64x1<f32>(64, 1)@(128, 128)
    isa(Aarch64Sme2)
    can_fuse(CAN_FUSE)

);

#[cfg(target_os = "macos")]
pub fn has_sme() -> bool {
    // TRACT_SME_DISABLE=1 forces the SME path off so callers can A/B
    // against the AMX path on the same binary.
    if crate::knobs::TRACT_SME_DISABLE.get() {
        return false;
    }
    // hw.optional.arm.FEAT_SME is an INTEGER sysctl, not a string. The
    // generic apple_get_syscall reads bytes-as-C-string which fails here
    // (`\x01\x00\x00\x00` would compare against the ASCII "1"), so we
    // read it as a u64 directly.
    use std::ffi::{CString, c_char, c_int, c_void};
    use std::ptr::null_mut;
    unsafe extern "C" {
        fn sysctlbyname(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *mut c_void,
            newlen: usize,
        ) -> c_int;
    }
    let Ok(name) = CString::new("hw.optional.arm.FEAT_SME") else {
        return false;
    };
    let mut value: u64 = 0;
    let mut len: usize = std::mem::size_of::<u64>();
    unsafe {
        if sysctlbyname(name.as_ptr(), &mut value as *mut _ as *mut c_void, &mut len, null_mut(), 0)
            != 0
        {
            return false;
        }
    }
    // FEAT_SME present AND the streaming vector length matches our kernels'
    // hardcoded 512-bit geometry.
    value != 0 && sme_geometry_supported()
}

#[cfg(target_os = "linux")]
pub fn has_sme() -> bool {
    // HWCAP2_SME = 1 << 23 on aarch64 (kernel ABI).
    const HWCAP2_SME: u64 = 1 << 23;
    unsafe extern "C" {
        fn getauxval(t: u64) -> u64;
    }
    const AT_HWCAP2: u64 = 26;
    let feat = unsafe { (getauxval(AT_HWCAP2) & HWCAP2_SME) != 0 };
    // FEAT_SME present AND the streaming vector length matches our kernels'
    // hardcoded 512-bit geometry (rejects qemu-user, which advertises SME
    // with a non-512 SVL — the cause of the cross-compiled CI failures).
    feat && sme_geometry_supported()
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub fn has_sme() -> bool {
    false
}

#[cfg(target_os = "macos")]
pub fn has_sme2() -> bool {
    // TRACT_SME_DISABLE=1 disables both SME and SME2 dispatch on the same
    // binary so end users can A/B the entire SME backend.
    if crate::knobs::TRACT_SME_DISABLE.get() {
        return false;
    }
    use std::ffi::{CString, c_char, c_int, c_void};
    use std::ptr::null_mut;
    unsafe extern "C" {
        fn sysctlbyname(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *mut c_void,
            newlen: usize,
        ) -> c_int;
    }
    let Ok(name) = CString::new("hw.optional.arm.FEAT_SME2") else {
        return false;
    };
    let mut value: u64 = 0;
    let mut len: usize = std::mem::size_of::<u64>();
    unsafe {
        if sysctlbyname(name.as_ptr(), &mut value as *mut _ as *mut c_void, &mut len, null_mut(), 0)
            != 0
        {
            return false;
        }
    }
    // FEAT_SME2 present AND the streaming vector length matches our kernels'
    // hardcoded 512-bit geometry.
    value != 0 && sme_geometry_supported()
}

#[cfg(target_os = "linux")]
pub fn has_sme2() -> bool {
    // HWCAP2_SME2 = 1 << 37 on aarch64 (kernel ABI).
    const HWCAP2_SME2: u64 = 1 << 37;
    unsafe extern "C" {
        fn getauxval(t: u64) -> u64;
    }
    const AT_HWCAP2: u64 = 26;
    let feat = unsafe { (getauxval(AT_HWCAP2) & HWCAP2_SME2) != 0 };
    // FEAT_SME2 present AND the streaming vector length matches our kernels'
    // hardcoded 512-bit geometry (rejects qemu-user, which advertises SME2
    // with a non-512 SVL — the cause of the cross-compiled CI failures).
    feat && sme_geometry_supported()
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub fn has_sme2() -> bool {
    false
}

fn sme_preferred(
    _isa: &IsaSet,
    dt: DatumType,
    query: &Query,
    _suitable: &[Suitable],
) -> Option<&'static str> {
    match (dt, query.n) {
        (DatumType::F32, Some(1)) => None,
        (DatumType::F32, _) => Some(sme_mmm_f32_32x32.name.as_str()),
        _ => None,
    }
}

fn sme2_preferred(
    _isa: &IsaSet,
    dt: DatumType,
    query: &Query,
    _suitable: &[Suitable],
) -> Option<&'static str> {
    match (dt, query.n) {
        (DatumType::F32, Some(1)) => Some(sme_mmv_f32_64x1.name.as_str()),
        (DatumType::I32, Some(1)) => None,
        (DatumType::I32, _) => Some(sme_qmmm_i32_32x32.name.as_str()),
        _ => None,
    }
}

inventory::submit! {
    MmmTier {
        arch: Some(crate::isa::Arch::Aarch64),
        precedence: 4,
        name: "sme",
        applies: |isa| isa.has(crate::isa::Isa::Aarch64Sme),
        preferred: sme_preferred,
    }
}

inventory::submit! {
    MmmTier {
        arch: Some(crate::isa::Arch::Aarch64),
        precedence: 5,
        name: "sme2",
        applies: |isa| isa.has(crate::isa::Isa::Aarch64Sme2),
        preferred: sme2_preferred,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::mmm::tests::packed_packed::PackedPackedProblem;
    use tract_data::internal::Approximation;
    use tract_data::internal::TractResult;

    // Phase 1A correctness: AddMatMul + Clear + Store + Done on a few
    // shapes. Bypasses auto-tests (SME_OFF) by calling run/reference
    // directly. Skipped if hardware lacks SME.
    fn check_shape(m_tile: usize, k: usize, n_tile: usize) {
        const MR: usize = 32;
        const NR: usize = 32;
        let m = m_tile * MR;
        let n = n_tile * NR;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.013) - 1.5).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.017) + 0.25).collect();
        let pb = PackedPackedProblem::kernel(&*sme_mmm_f32_32x32, 0, a, b);
        let expected = pb.reference().expect("scalar reference");
        let found = pb.run().expect("SME kernel run");
        found
            .close_enough(&expected, Approximation::Approximate)
            .unwrap_or_else(|e| panic!("SME mmm mismatch at k={k}: {e}"));
    }

    #[test]
    fn sme_mmm_f32_32x32_k1() {
        if !has_sme() {
            eprintln!("SME not present, skipping");
            return;
        }
        check_shape(1, 1, 1);
    }

    #[test]
    fn sme_mmm_f32_32x32_k8() {
        if !has_sme() {
            return;
        }
        check_shape(1, 8, 1);
    }

    #[test]
    fn sme_mmm_f32_32x32_k128() {
        if !has_sme() {
            return;
        }
        check_shape(1, 128, 1);
    }

    #[test]
    fn sme_mmm_f32_32x32_multi_tile() {
        if !has_sme() {
            return;
        }
        // 64x64 output (2x2 tiles), K=64 — exercises the framework
        // iterating across multiple kernel calls.
        check_shape(2, 64, 2);
    }

    #[test]
    fn sme_qmmm_i8_output() -> TractResult<()> {
        if !has_sme2() {
            return Ok(());
        }
        let a = vec![7.0; 32 * 8];
        let b = vec![5.0; 8 * 32];
        PackedPackedProblem::kernel(&*sme_qmmm_i32_32x32, 1, a, b)
            .with_output_type(DatumType::I8)
            .check()
    }

    #[test]
    fn sme_qmmm_i8_output_partial_tiles() -> TractResult<()> {
        if !has_sme2() {
            return Ok(());
        }
        let a: Vec<f32> = (0..35 * 8).map(|i| (i % 5) as f32 - 2.0).collect();
        let b: Vec<f32> = (0..8 * 37).map(|i| (i % 7) as f32 - 3.0).collect();
        PackedPackedProblem::frame(&*sme_qmmm_i32_32x32, 1, 35, 37, a, b)
            .with_output_type(DatumType::I8)
            .check()
    }

    // Output laid out [n, m], so M is contiguous and N is strided. That is the
    // recognition matmul layout, and it has to match a row-major reference.
    #[test]
    fn sme_matmul_stores_along_m() {
        if !has_sme() {
            return;
        }
        use crate::mmm::{AsInputValue, FusedSpec};
        for (m, k, n) in [(32usize, 8usize, 32usize), (40, 8, 40), (48, 4, 96), (12, 8, 32)] {
            let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.013) - 1.5).collect();
            let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.017) + 0.25).collect();
            let problem = PackedPackedProblem::frame(&*sme_mmm_f32_32x32, 0, m, n, a, b);
            let expected = problem.reference().unwrap();
            let (packed_a, packed_b) = problem.padded_inputs().unwrap();
            let (pack_a, pack_b) =
                &crate::frame::mmm::kernel::MatMatMulKer::packings(&*sme_mmm_f32_32x32)[0];
            let pa = pack_a.prepare_one(&packed_a, 1, 0).unwrap();
            let pb = pack_b.prepare_one(&packed_b, 0, 1).unwrap();
            let mut found = tract_data::internal::Tensor::zero::<f32>(&[n, m]).unwrap();
            unsafe {
                let c = sme_mmm_f32_32x32.c_view(Some(1), Some(0)).wrap(&found.view_mut());
                sme_mmm_f32_32x32
                    .run(
                        m,
                        n,
                        &[
                            FusedSpec::AddMatMul {
                                a: AsInputValue::Borrowed(&*pa),
                                b: AsInputValue::Borrowed(&*pb),
                                packing: 0,
                            },
                            FusedSpec::Store(c),
                        ],
                    )
                    .unwrap();
            }
            let exp = expected.to_plain_array_view::<f32>().unwrap();
            let got = found.to_plain_array_view::<f32>().unwrap();
            for mi in 0..m {
                for ni in 0..n {
                    let g = got[[ni, mi]];
                    let e = exp[[mi, ni]];
                    assert!((g - e).abs() < 1e-3, "{m}x{k}x{n} ({mi},{ni}) got {g} expected {e}");
                }
            }
        }
    }

    // LoadTile writes a known pattern. Row-major hits the contiguous-N store,
    // column-major hits the vertical-slice store, arbitrary strides stay on
    // the scalar path.
    #[test]
    fn sme_store_layouts() {
        if !has_sme() {
            return;
        }
        use crate::frame::mmm::tests::store::{StoreLayout, store_pattern};
        for layout in [StoreLayout::RowMajor, StoreLayout::ColMajor, StoreLayout::Arbitrary] {
            store_pattern::<_, f32, f32>(&*sme_mmm_f32_32x32, layout);
        }
    }

    // m <= 16 on a full N panel stores those rows directly. m > 16, a short
    // N panel, a full 32-row panel, and strided A stay on the old tile walk.
    // Guard rows past m must stay untouched, including with the fused
    // row-add + scalar-max that Relu.1.low runs.
    fn check_live_rows(packing: usize, m: usize, k: usize, n: usize, fuse: bool) {
        use crate::BinOp;
        use crate::mmm::{AsInputValue, FusedSpec};
        let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.013) - 1.5).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.017) + 0.25).collect();
        let bias: Vec<f32> = (0..m).map(|i| (i as f32) * 0.1 - 0.3).collect();
        let cap = 0.25f32;
        let problem = PackedPackedProblem::frame(
            &*sme_mmm_f32_32x32,
            packing,
            m,
            n,
            a_data.clone(),
            b_data.clone(),
        );
        let (packed_a, packed_b) = problem.padded_inputs().unwrap();
        let (pack_a, pack_b) =
            &crate::frame::mmm::kernel::MatMatMulKer::packings(&*sme_mmm_f32_32x32)[packing];
        let pa = pack_a.prepare_one(&packed_a, 1, 0).unwrap();
        let pb = pack_b.prepare_one(&packed_b, 0, 1).unwrap();
        let rows = m + 32;
        let storage = vec![7.5f32; rows * n];
        let mut found = tract_data::internal::Tensor::from_shape(&[rows, n], &storage).unwrap();
        let bias_t = tract_data::internal::Tensor::from_shape(&[m], &bias).unwrap();
        let cap_t = tract_data::internal::tensor0(cap);
        unsafe {
            let c = sme_mmm_f32_32x32.c_view(Some(0), Some(1)).wrap(&found.view_mut());
            let mut ops = vec![FusedSpec::AddMatMul {
                a: AsInputValue::Borrowed(&*pa),
                b: AsInputValue::Borrowed(&*pb),
                packing,
            }];
            if fuse {
                ops.push(FusedSpec::BinPerRow(bias_t.view(), BinOp::Add));
                ops.push(FusedSpec::BinScalar(&cap_t, BinOp::Max));
            }
            ops.push(FusedSpec::Store(c));
            sme_mmm_f32_32x32.run(m, n, &ops).unwrap();
        }
        let got = found.to_plain_array_view::<f32>().unwrap();
        for mi in 0..m {
            for ni in 0..n {
                let mut acc = 0f32;
                for ki in 0..k {
                    acc += a_data[ki + k * mi] * b_data[ni + n * ki];
                }
                if fuse {
                    acc = (acc + bias[mi]).max(cap);
                }
                let g = got[[mi, ni]];
                let tol = 1e-4 * acc.abs().max(1.0);
                assert!(
                    (g - acc).abs() <= tol,
                    "pack {packing} fuse {fuse} {m}x{k}x{n} ({mi},{ni}) got {g} expected {acc}"
                );
            }
        }
        for mi in m..rows {
            for ni in 0..n {
                let g = got[[mi, ni]];
                assert_eq!(g, 7.5, "wrote past live m at ({mi},{ni}): {g}");
            }
        }
    }

    #[test]
    fn sme_partial_m_rows() {
        if !has_sme() {
            return;
        }
        for fuse in [false, true] {
            for (m, k, n) in
                [(1usize, 4usize, 32usize), (12, 96, 64), (12, 96, 40), (16, 8, 32), (16, 3, 96)]
            {
                check_live_rows(0, m, k, n, fuse);
            }
        }
        for (m, k, n) in [
            (17usize, 8usize, 32usize),
            (24, 8, 64),
            (32, 8, 32),
            (40, 8, 64),
            (48, 4, 32),
            (64, 4, 64),
        ] {
            check_live_rows(0, m, k, n, true);
        }
    }

    // Strided store path: hand-built Clear + Store chain with non-contig C.
    #[test]
    fn sme_store_non_contiguous() {
        if !has_sme() {
            return;
        }
        use crate::frame::mmm::{FusedKerSpec, OutputStoreKer};
        const MR: usize = 32;
        const NR: usize = 32;
        let mut v: Vec<f32> = vec![f32::MAX; MR * 5 * NR * 3];
        let c = OutputStoreKer {
            ptr: v.as_mut_ptr() as _,
            row_byte_stride: (4 * 3 * NR * 5) as isize,
            col_byte_stride: 4 * 3,
            item_size: 4,
        };
        let non_linear = [FusedKerSpec::<f32>::Clear, FusedKerSpec::Store(c), FusedKerSpec::Done];
        let err = unsafe { (sme_mmm_f32_32x32.kernel)(&non_linear) };
        assert_eq!(err, 0, "kernel returned non-zero error code");
        let mut expected = vec![f32::MAX; v.len()];
        for col in 0..NR {
            for row in 0..MR {
                expected[col * 3 + row * 3 * 5 * NR] = 0.0;
            }
        }
        for (i, (got, exp)) in v.iter().zip(expected.iter()).enumerate() {
            assert_eq!(got, exp, "mismatch at idx {i}: got {got} expected {exp}");
        }
    }
}
