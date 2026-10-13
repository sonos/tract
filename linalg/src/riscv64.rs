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
//! computes a short tile when `VLMAX < MR`. Each kernel therefore declares the
//! narrowest vector unit that reaches its `MR`: [`Isa::RiscV64V`] for the
//! `VLEN >= 128` every RVV 1.0 hart mandates, [`Isa::RiscV64Vlen256`] above it.
//! The element-wise and reduction kernels have no such constraint -- they
//! strip-mine on `vsetvli` and take whatever `vl` the hart grants -- so they
//! declare the vector unit and nothing more.
//! `SEW` is the other half of `VLMAX`, so the f16 tiles are twice as tall as
//! the f32 ones at the same `LMUL` and reuse those same two widths, with
//! [`Isa::RiscV64Zvfh`] on top for the arithmetic itself.
//!
//! Everything here rests on [`has_rvv`] being true only for the *ratified*
//! extension: the 0.7.1 draft parts share neither encodings nor CSRs, and the
//! kernels are not merely slow there but wrong.

use crate::isa::{Arch, Isa, IsaSet};

// The element-wise and reduction kernels are Rust `asm!` blocks, so rustc's own
// assembler is the only one they need. The matmul kernels are `.S` files and exist only where
// build.rs found an external assembler that could encode RVV 1.0 -- which is what `tract_rvv`
// records, and why they alone are gated on it.
mod by_scalar;
mod reduce;
#[cfg(tract_rvv)]
mod rvv;
mod unicast;
#[cfg(tract_rvv)]
pub use rvv::*;

/// `__NR_riscv_hwprobe`, Linux 6.4 and later.
#[cfg(all(target_arch = "riscv64", target_os = "linux"))]
const NR_RISCV_HWPROBE: libc::c_long = 258;

/// `RISCV_HWPROBE_KEY_MVENDORID`.
const HWPROBE_KEY_MVENDORID: i64 = 0;

/// `RISCV_HWPROBE_KEY_MARCHID`.
const HWPROBE_KEY_MARCHID: i64 = 1;

/// `RISCV_HWPROBE_KEY_IMA_EXT_0`, the extension bitmap.
const HWPROBE_KEY_IMA_EXT_0: i64 = 4;

/// SpacemiT's JEDEC vendor id and the X60 core's architecture id, as found on
/// the K1 / Ky X1 SoCs.
const SPACEMIT_MVENDORID: u64 = 0x710;
const SPACEMIT_X60_MARCHID: u64 = 0x8000_0000_5800_0001;

/// Matmul size, in multiply-adds, from which the X60 gets a TLB flush ahead of
/// the run: below it the flush costs more than the run can lose.
const X60_TLB_REFRESH_FMAS: usize = 8 << 20;

/// `RISCV_HWPROBE_IMA_V`, which the kernel sets only for the ratified vector
/// extension.
const HWPROBE_IMA_V: u64 = 1 << 2;

/// `RISCV_HWPROBE_EXT_ZVFH`. Zvfh, not Zvfhmin: the latter offers only
/// f16<->f32 conversion and so cannot carry an f16 accumulator. RVA23 mandates
/// Zvfhmin and leaves Zvfh optional, so the profile is not enough to infer it.
const HWPROBE_EXT_ZVFH: u64 = 1 << 30;

/// `struct riscv_hwprobe`.
#[cfg(all(target_arch = "riscv64", target_os = "linux"))]
#[repr(C)]
struct HwprobePair {
    key: i64,
    value: u64,
}

/// One `riscv_hwprobe(2)` key, or `None` where the kernel cannot answer it.
#[cfg(all(target_arch = "riscv64", target_os = "linux"))]
fn hwprobe(key: i64) -> Option<u64> {
    let mut pair = HwprobePair { key, value: 0 };
    // SAFETY: the syscall writes only through the pointer we hand it, to one
    // pair of the layout it expects. A kernel without it fails with ENOSYS and
    // writes nothing, leaving `pair` as initialised here.
    let rc = unsafe {
        libc::syscall(
            NR_RISCV_HWPROBE,
            &mut pair as *mut HwprobePair,
            1 as libc::c_ulong,
            0 as libc::c_ulong,
            std::ptr::null_mut::<libc::c_ulong>(),
            0 as libc::c_uint,
        )
    };
    // An unrecognised key comes back as -1 with the value left at 0.
    (rc == 0 && pair.key == key).then_some(pair.value)
}

#[cfg(not(all(target_arch = "riscv64", target_os = "linux")))]
fn hwprobe(_key: i64) -> Option<u64> {
    None
}

/// The extension bitmap, asked through `riscv_hwprobe(2)`; 0 where it cannot
/// be asked.
///
/// The `AT_HWCAP` 'V' bit cannot answer this. Linux derives that bitmap from
/// the ISA string the firmware reports, and a T-Head C910 — BeagleV-Ahead,
/// Sipeed LicheePi 4A — advertises a bare `v` while implementing the
/// incompatible 0.7.1 draft, so the bit is set on a hart that faults on, or
/// silently misdecodes, every 1.0 encoding. `hwprobe` is the interface that
/// distinguishes them, and a kernel too old to have it is treated as having no
/// vector unit: the draft parts are the old-kernel population, and losing the
/// kernels on a 1.0 hart running an old kernel costs speed, not correctness.
/// It is also the only interface carrying the multi-letter extensions —
/// `AT_HWCAP` has bits for the single-letter ones alone, so Zvfh has no answer
/// there at all.
fn probe_ima_ext_0() -> u64 {
    hwprobe(HWPROBE_KEY_IMA_EXT_0).unwrap_or(0)
}

/// Reads `vlenb`, the read-only CSR holding `VLEN / 8`.
///
/// Callers must establish [`has_rvv`] first: without vector state the CSR read
/// raises an illegal instruction, which arrives as SIGILL and cannot be
/// recovered from.
#[cfg(target_arch = "riscv64")]
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

#[cfg(not(target_arch = "riscv64"))]
fn read_vlenb() -> usize {
    0
}

lazy_static::lazy_static! {
    static ref IMA_EXT_0: u64 = probe_ima_ext_0();

    static ref HAS_RVV: bool = *IMA_EXT_0 & HWPROBE_IMA_V != 0;

    static ref HAS_ZVFH: bool = *HAS_RVV && *IMA_EXT_0 & HWPROBE_EXT_ZVFH != 0;

    static ref VLENB: usize = if *HAS_RVV { read_vlenb() } else { 0 };

    static ref IS_SPACEMIT_X60: bool = hwprobe(HWPROBE_KEY_MVENDORID) == Some(SPACEMIT_MVENDORID)
        && hwprobe(HWPROBE_KEY_MARCHID) == Some(SPACEMIT_X60_MARCHID);

    static ref TLB_REFRESH_FMAS: Option<usize> =
        match crate::knobs::TRACT_RISCV64_TLB_REFRESH_FMAS.get() {
            Some(0) => None,
            Some(n) => Some(n),
            None => IS_SPACEMIT_X60.then_some(X60_TLB_REFRESH_FMAS),
        };
}

/// Flush the TLB ahead of a matmul of `fmas` multiply-adds, on the cores that
/// need it.
///
/// The SpacemiT X60's L1 data prefetcher drifts, over a run without TLB
/// maintenance, into a state where it stops covering the matmul operand
/// streams, and stays there until the TLB is flushed. A process taking page
/// faults gets those flushes as a side effect; one recycling its memory does
/// not. User code cannot issue `sfence.vma`, so this changes the protection of
/// a private page, which makes the kernel flush on every hart running the
/// process. Matmuls under the threshold are left alone. The threshold comes
/// from `TRACT_RISCV64_TLB_REFRESH_FMAS`, read once.
pub fn refresh_tlb_before_matmul(fmas: usize) {
    if TLB_REFRESH_FMAS.is_some_and(|threshold| fmas >= threshold) {
        flush_tlb();
    }
}

#[cfg(all(target_arch = "riscv64", target_os = "linux"))]
fn flush_tlb() {
    use std::sync::atomic::{AtomicBool, Ordering};
    static PAGE: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    static WRITABLE: AtomicBool = AtomicBool::new(true);
    let Some(page) = *PAGE.get_or_init(|| {
        // SAFETY: a fresh private anonymous page, owned by this static for the
        // life of the process and never handed out. It is written once so a PTE
        // exists: changing the protection of an unpopulated page flushes nothing.
        unsafe {
            let p = libc::mmap(
                std::ptr::null_mut(),
                4096,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            );
            if p == libc::MAP_FAILED {
                return None;
            }
            (p as *mut u8).write_volatile(1);
            Some(p as usize)
        }
    }) else {
        return;
    };
    let prot = if WRITABLE.fetch_xor(true, Ordering::Relaxed) {
        libc::PROT_READ
    } else {
        libc::PROT_READ | libc::PROT_WRITE
    };
    // SAFETY: the page is ours and nothing reads or writes it after setup.
    unsafe { libc::mprotect(page as *mut libc::c_void, 4096, prot) };
}

#[cfg(not(all(target_arch = "riscv64", target_os = "linux")))]
fn flush_tlb() {}

/// Whether the hart implements the ratified RVV 1.0 vector extension.
pub fn has_rvv() -> bool {
    *HAS_RVV
}

/// Whether the hart implements Zvfh, f16 arithmetic in the vector unit.
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

/// As [`vlmax_f32`], for 16-bit elements: half the `SEW`, so twice the lanes.
pub fn vlmax_f16(lmul: usize) -> usize {
    vlenb() * lmul / std::mem::size_of::<crate::f16>()
}

/// What this hart has, in the shared vocabulary. `VLEN` is not an instruction
/// set feature, but it decides which tile heights are reachable, so the width
/// the wide kernels need is carried as a step of its own.
pub fn isa_set() -> IsaSet {
    let mut set = IsaSet::of_arch(Arch::RiscV64);
    if has_rvv() {
        set = set.with(Isa::RiscV64V);
        if vlenb() >= 32 {
            set = set.with(Isa::RiscV64Vlen256);
        }
        if has_zvfh() {
            set = set.with(Isa::RiscV64Zvfh);
        }
        log::info!("RVV 1.0 available, VLEN = {} bits, zvfh = {}", vlenb() * 8, has_zvfh());
    }
    set
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
            assert_eq!(vlmax_f16(1), 2 * vlmax_f32(1));
        } else {
            assert_eq!(vlenb(), 0);
            assert!(!has_zvfh(), "Zvfh is a vector extension and cannot be present without V");
        }
    }

    /// The vector width the wide kernels declare must be the one they actually
    /// need: `VLMAX >= MR` for the `(MR, LMUL)` build.rs rendered them from.
    #[test]
    fn vlen256_is_what_the_wide_tiles_need() {
        let set = isa_set();
        assert_eq!(set.has(Isa::RiscV64Vlen256), has_rvv() && vlmax_f32(2) >= 16);
        assert_eq!(set.has(Isa::RiscV64Vlen256), has_rvv() && vlmax_f32(8) >= 64);
        assert_eq!(set.has(Isa::RiscV64Vlen256), has_rvv() && vlmax_f16(2) >= 32);
        assert_eq!(set.has(Isa::RiscV64Vlen256), has_rvv() && vlmax_f16(8) >= 128);
    }
}
