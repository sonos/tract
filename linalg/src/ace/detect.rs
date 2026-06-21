//! Runtime detection of ACE support.
//!
//! `has_ace()` returns **false on all current hardware and on non-x86** — there is
//! no ACE silicon (expected ~2028) and no published CPUID assignment yet. It exists
//! so the future real-asm kernel (the `cfg(tract_ace)` sibling of the emulation) can
//! gate on it the way the AMX kernels gate on `has_amx_int8()`. It is deliberately
//! NOT used to gate the *emulated* kernels — doing so would make their differential
//! tests silently no-op on hosts without ACE.

#[cfg(target_arch = "x86_64")]
mod imp {
    use std::sync::OnceLock;

    // PLACEHOLDER feature location. The exact ACE CPUID leaf/sub-leaf/bit is
    // unpublished. These point at a currently-RESERVED bit so the probe reads 0 on
    // every shipping CPU. TODO(ace): replace with the assigned leaf/bit (and split
    // int8 / bf16 / MX sub-capabilities) once the ACE ISA reference is public.
    const ACE_CPUID_LEAF: u32 = 7;
    const ACE_CPUID_SUBLEAF: u32 = 1;
    const ACE_CPUID_BIT_EAX: u32 = 1 << 0; // reserved today -> reads 0 everywhere

    fn cpu_has_ace() -> bool {
        if !std::is_x86_feature_detected!("avx512f") {
            return false;
        }
        #[allow(unused_unsafe)]
        let r = unsafe { std::arch::x86_64::__cpuid_count(ACE_CPUID_LEAF, ACE_CPUID_SUBLEAF) };
        (r.eax & ACE_CPUID_BIT_EAX) != 0
    }

    // ACE is revealed "as a new palette under the AMX framework", so its tile +
    // Block Scale Register state lives in the AMX XSAVE family. Reuse the AMX
    // XFEATURE_XTILEDATA XCOMP-perm gate. TODO(ace): the Block Scale Register may
    // need a distinct XFEATURE beyond XTILEDATA — add a second arch_prctl if so.
    #[cfg(target_os = "linux")]
    fn request_ace_xcomp_perm() -> bool {
        // arch_prctl(ARCH_REQ_XCOMP_PERM=0x1023, XFEATURE_XTILEDATA=18) -> 0 on ok.
        let rc: i64;
        unsafe {
            std::arch::asm!(
                "syscall",
                in("rax") 158i64,
                in("rdi") 0x1023i64,
                in("rsi") 18i64,
                lateout("rax") rc,
                out("rcx") _,
                out("r11") _,
                options(nostack),
            );
        }
        rc == 0
    }
    #[cfg(not(target_os = "linux"))]
    fn request_ace_xcomp_perm() -> bool {
        false
    }

    pub fn has_ace() -> bool {
        static GATE: OnceLock<bool> = OnceLock::new();
        *GATE.get_or_init(|| cpu_has_ace() && request_ace_xcomp_perm())
    }
}

#[cfg(not(target_arch = "x86_64"))]
mod imp {
    pub fn has_ace() -> bool {
        false
    }
}

pub use imp::has_ace;

#[cfg(test)]
mod tests {
    #[test]
    fn has_ace_is_false_today() {
        // No ACE silicon exists; the placeholder CPUID bit is reserved (reads 0),
        // and non-x86 hosts return false. Must never panic.
        assert!(!super::has_ace());
    }
}
