#![allow(dead_code)]

#[cfg(not(target_arch = "aarch64"))]
pub use generic::*;

#[cfg(target_arch = "aarch64")]
pub use aarch64::*;

mod generic {
    pub fn now() -> std::time::Instant {
        std::time::Instant::now()
    }
}

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use std::arch::asm;
    use std::time::Duration;

    pub struct Timestamp(u64);

    impl Timestamp {
        pub fn elapsed(&self) -> Duration {
            let diff = timestamp().saturating_sub(self.0) as f64;
            let secs = diff / frequency() as f64;
            std::time::Duration::from_secs_f64(secs)
        }
    }

    pub fn now() -> Timestamp {
        Timestamp(timestamp())
    }

    #[inline]
    fn frequency() -> u64 {
        unsafe {
            let frequency: u64;
            asm!(
                "mrs {}, cntfrq_el0",
                out(reg) frequency,
                options(nomem, nostack, preserves_flags, pure),
            );
            frequency
        }
    }

    #[inline(always)]
    fn timestamp() -> u64 {
        unsafe {
            let timestamp: u64;
            asm!(
                "mrs {}, cntvct_el0",
                out(reg) timestamp,
                // Leave off `nomem` because this should be a compiler fence.
                options(nostack, preserves_flags),
            );
            timestamp
        }
    }
}
