//! RISC-V (rv64) backend for the ratified Vector extension, RVV 1.0.
//!
//! Kernels are assembly rendered from jinja, as on x86_64 and arm64, because
//! Rust exposes no stable RVV intrinsics and `-C target-feature=+v` is itself
//! unstable.
//!
//! RVV is vector-length agnostic -- `VLEN` is a runtime property of the hart --
//! while `MR` and `NR` must be const generics, since they select the packing
//! format. Kernels reconcile the two by fixing `(MR, NR, LMUL)` and pinning
//! `vl` to `MR`. `vsetvli` clamps to `VLMAX`, so such a kernel is correct
//! wherever `VLMAX >= MR`, merely leaves lanes idle when `VLMAX > MR`, and
//! computes a short tile when `VLMAX < MR`. Every kernel is therefore gated on
//! [`vlmax_f32`] or [`vlmax_f16`] reaching its `MR`.

use crate::Ops;

// `tract_rvv` is set by build.rs only when the assembler could encode RVV 1.0;
// without it the kernel symbols do not exist and dispatch stays generic.
#[cfg(tract_rvv)]
mod rvv;
#[cfg(tract_rvv)]
pub use rvv::*;

/// `AT_HWCAP` -- see `getauxval(3)`.
const AT_HWCAP: libc::c_ulong = 16;

/// Bit `'V' - 'A'` of the single-letter extension bitmap Linux puts in
/// `AT_HWCAP` (`COMPAT_HWCAP_ISA_V`).
///
/// This bit alone is a sufficient RVV 1.0 gate: Linux sets it only for the
/// ratified extension, never for the incompatible 0.7.1 draft implemented by
/// the Allwinner D1 and Sophgo SG2042.
const HWCAP_ISA_V: libc::c_ulong = 1 << (b'V' - b'A');

fn hwcap() -> libc::c_ulong {
    // SAFETY: getauxval is thread-safe and takes a scalar; it returns 0 for an
    // unknown type, which reads here as "no vector unit".
    unsafe { libc::getauxval(AT_HWCAP) }
}

/// Reads `vlenb`, the read-only CSR holding `VLEN / 8`.
///
/// Callers must establish [`has_rvv`] first: without vector state the CSR read
/// raises an illegal instruction, which arrives as SIGILL and cannot be
/// recovered from.
fn read_vlenb() -> usize {
    let vlenb: usize;
    // SAFETY: guarded by has_rvv(). `csrr` on a read-only CSR has no side
    // effects. The CSR is named numerically so the assembler does not need to
    // know about vector extensions to encode it.
    unsafe {
        std::arch::asm!("csrr {out}, 0xC22", out = out(reg) vlenb, options(nomem, nostack, preserves_flags));
    }
    vlenb
}

/// Splits the kernel-canonicalised `isa` line of /proc/cpuinfo, e.g.
/// `rv64imafdcv_zicsr_zvfh_zvl256b`, into lowercase extension tokens.
///
/// This is the only source for multi-letter extensions; `AT_HWCAP` carries
/// bits for the single-letter ones alone.
fn isa_extensions() -> Vec<String> {
    #[cfg(test)]
    crate::setup_test_logger();
    let Ok(cpu_info) = std::fs::read_to_string("/proc/cpuinfo") else {
        log::warn!("Could not read /proc/cpuinfo. CPU feature detection may be impaired.");
        return vec![];
    };
    let Some(line) = cpu_info.lines().find(|line| line.trim_start().starts_with("isa")) else {
        log::warn!("No \"isa :\" line in /proc/cpuinfo. CPU feature detection may be impaired.");
        return vec![];
    };
    let Some((_, isa)) = line.split_once(':') else { return vec![] };
    isa.trim().split('_').map(|s| s.to_lowercase()).collect()
}

lazy_static::lazy_static! {
    static ref HAS_RVV: bool = hwcap() & HWCAP_ISA_V != 0;

    static ref VLENB: usize = if *HAS_RVV { read_vlenb() } else { 0 };

    /// Zvfh, not Zvfhmin: the latter offers only f16<->f32 conversion and so
    /// cannot carry an f16 accumulator. RVA23 mandates Zvfhmin and leaves Zvfh
    /// optional, so the profile is not enough to infer it.
    static ref HAS_ZVFH: bool = *HAS_RVV && isa_extensions().iter().any(|e| e == "zvfh");
}

/// Whether the hart implements the ratified RVV 1.0 vector extension.
pub fn has_rvv() -> bool {
    *HAS_RVV
}

/// Whether the hart implements Zvfh (native f16 vector arithmetic).
pub fn has_zvfh() -> bool {
    *HAS_ZVFH
}

/// Vector register width in bytes (`VLEN / 8`); 0 without RVV.
pub fn vlenb() -> usize {
    *VLENB
}

/// `VLMAX = LMUL * VLEN / SEW` for 32-bit elements: the largest `vl` this hart
/// can grant. A kernel of tile height `MR` is dispatchable only where this
/// reaches `MR`.
pub fn vlmax_f32(lmul: usize) -> usize {
    vlenb() * lmul / std::mem::size_of::<f32>()
}

/// As [`vlmax_f32`], for 16-bit elements.
pub fn vlmax_f16(lmul: usize) -> usize {
    vlenb() * lmul / std::mem::size_of::<crate::f16>()
}

pub fn plug(_ops: &mut Ops) {
    if has_rvv() {
        #[cfg(tract_rvv)]
        rvv::plug(_ops);
    }
}

#[cfg(test)]
mod test {
    use super::*;

    /// Detection must be internally consistent and must not trap. Written to
    /// pass on a hart without V as well, so it stays meaningful under
    /// qemu-riscv64 with and without `v=true`.
    #[test]
    fn detection_is_coherent() {
        eprintln!(
            "rvv={} VLEN={} zvfh={} vlmax_f32(lmul=1,2,4)={:?}",
            has_rvv(),
            vlenb() * 8,
            has_zvfh(),
            [vlmax_f32(1), vlmax_f32(2), vlmax_f32(4)],
        );
        if has_rvv() {
            let vlenb = vlenb();
            assert!(vlenb >= 16, "RVV 1.0 mandates VLEN >= 128, got VLEN={}", vlenb * 8);
            assert!(vlenb.is_power_of_two(), "VLEN must be a power of two, got {}", vlenb * 8);
            assert_eq!(vlmax_f32(1), vlenb / 4);
            assert_eq!(vlmax_f32(4), vlenb);
        } else {
            assert_eq!(vlenb(), 0);
            assert!(!has_zvfh(), "Zvfh cannot be present without V");
        }
    }
}
