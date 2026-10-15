//! Runtime detection of ACE v1 support.
//!
//! [`has_ace`] follows the detection algorithm of the ACE specification (§3.2):
//! AVX10.1 + AVX10_V1_AUX (or AVX10.2), AVX10_V2_AUX, the ACE feature flag,
//! `ACE_VSN >= 1`, OS-enabled XSAVE state for TILECFG/TILEDATA/SCALEDATA and
//! AVX-512, and — on Linux — the XTILEDATA permission from `arch_prctl` (IA32_XFD
//! only gates bit 18; SCALEDATA has no XFD bit). No shipping CPU passes it; Intel
//! SDE `-future-ag` does.

#[cfg(target_arch = "x86_64")]
mod imp {
    use std::arch::x86_64::{__cpuid_count, CpuidResult};
    use std::sync::OnceLock;

    /// `__cpuid_count` is an `unsafe fn` on the MSRV toolchain but safe on
    /// current stable.
    #[allow(unused_unsafe)]
    pub(super) fn cpuid(leaf: u32, sub_leaf: u32) -> CpuidResult {
        unsafe { __cpuid_count(leaf, sub_leaf) }
    }

    fn bit(x: u32, b: u32) -> bool {
        (x >> b) & 1 == 1
    }

    fn xcr0() -> u64 {
        let (lo, hi): (u32, u32);
        unsafe {
            std::arch::asm!("xgetbv", in("ecx") 0, out("eax") lo, out("edx") hi,
                options(nomem, nostack, preserves_flags));
        }
        ((hi as u64) << 32) | lo as u64
    }

    fn cpu_has_ace() -> bool {
        let max_leaf = cpuid(0, 0).eax;
        if max_leaf < 0x24 || !bit(cpuid(1, 0).ecx, 27) {
            return false;
        }
        let l7_1 = cpuid(7, 1);
        let (avx10, ace) = (bit(l7_1.edx, 19), bit(l7_1.ecx, 11));
        if !avx10 || !ace || !bit(cpuid(7, 0).edx, 24) {
            return false;
        }
        let avx10_vsn = cpuid(0x24, 0).ebx & 0xFF;
        let aux = cpuid(0x24, 1).ecx;
        let avx10_ok = (avx10_vsn >= 1 && bit(aux, 2)) || avx10_vsn >= 2;
        let palette_ok = cpuid(0x1D, 0).eax >= 2;
        let ace_vsn = cpuid(0x1D, 2).eax & 0xFF;
        // XCR0: x87|SSE|AVX (2:1), opmask|ZMM_Hi256|Hi16_ZMM (7:5), TILECFG|TILEDATA
        // (18:17), SCALEDATA (20).
        const XCR0_ACE: u64 = 0b110 | (0b111 << 5) | (0b11 << 17) | (1 << 20);
        avx10_ok && bit(aux, 3) && palette_ok && ace_vsn >= 1 && xcr0() & XCR0_ACE == XCR0_ACE
    }

    #[cfg(target_os = "linux")]
    fn request_xtiledata_perm() -> bool {
        // arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)
        let rc: i64;
        unsafe {
            std::arch::asm!(
                "syscall",
                inlateout("rax") 158i64 => rc,
                in("rdi") 0x1023i64,
                in("rsi") 18i64,
                out("rcx") _,
                out("r11") _,
                options(nostack),
            );
        }
        rc == 0
    }

    #[cfg(not(target_os = "linux"))]
    fn request_xtiledata_perm() -> bool {
        false
    }

    pub fn has_ace() -> bool {
        static GATE: OnceLock<bool> = OnceLock::new();
        *GATE.get_or_init(|| cpu_has_ace() && request_xtiledata_perm())
    }
}

#[cfg(not(target_arch = "x86_64"))]
mod imp {
    pub fn has_ace() -> bool {
        false
    }
}

pub use imp::has_ace;

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    #[test]
    fn has_ace_implies_the_ace_cpuid_flag() {
        let flag = (super::imp::cpuid(7, 1).ecx >> 11) & 1 == 1;
        assert!(!super::has_ace() || flag);
    }
}
