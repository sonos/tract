use crate::Ops;
use crate::frame::mmm::ImplementationQuality::ManuallyOptimized;
use crate::mmm::*;

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

const SME: fn() -> bool = has_sme;
const SME2: fn() -> bool = has_sme2;
// The SMOPA i32 kernel implements the quant fuse ops (QScale / RoundingShiftRight
// / ShiftLeft) bit-exactly; only LeakyRelu is unsupported (kernel returns 1).
const CAN_FUSE_I32: fn(&FusedSpec) -> bool = |f| !matches!(f, FusedSpec::LeakyRelu(_));

MMMExternKernel!(sme_qmmm_i32_32x32<i32>(32,32)@(128,128) where(SME2) can_fuse(CAN_FUSE_I32)
    packing[1] = i8i8 => |k| k.with_packing(crate::pack::PackedI8K4::new(32), crate::pack::PackedI8K4::new(32));
    quality(ManuallyOptimized) store(i8));

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

MMMExternKernel!(
    sme_mmm_f32_32x32<f32>(32, 32)@(128, 128)
    where(SME)
    can_fuse(CAN_FUSE)
    quality(ManuallyOptimized)
);

MMMExternKernel!(
    sme_mmv_f32_64x1<f32>(64, 1)@(128, 128)
    where(SME2)
    can_fuse(CAN_FUSE)
    quality(ManuallyOptimized)
);

#[cfg(target_os = "macos")]
pub fn has_sme() -> bool {
    // TRACT_SME_DISABLE=1 forces the SME path off so callers can A/B
    // against the AMX path on the same binary.
    if std::env::var_os("TRACT_SME_DISABLE").is_some() {
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
    if std::env::var_os("TRACT_SME_DISABLE").is_some() {
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

pub fn plug(ops: &mut Ops) {
    if has_sme() {
        log::info!("SME optimisation activated");
        ops.mmm_f32 = Box::new(|_, _, _| sme_mmm_f32_32x32.mmm());
        ops.mmm_impls.extend_from_slice(&[sme_mmm_f32_32x32.mmm()]);
    }
    if has_sme2() {
        log::info!("SME2 GEMV optimisation activated");
        ops.mmv_f32 = Box::new(|_, _| sme_mmv_f32_64x1.mmm());
        ops.qmmm_i32 = Box::new(|_, _, _| sme_qmmm_i32_32x32.mmm());
        ops.mmm_impls.extend_from_slice(&[sme_mmv_f32_64x1.mmm(), sme_qmmm_i32_32x32.mmm()]);
    }
    if !has_sme() && !has_sme2() {
        log::info!("No SME optimisation");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::mmm::tests::packed_packed::PackedPackedProblem;
    use tract_data::internal::Approximation;

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
