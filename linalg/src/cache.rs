//! Best-effort runtime CPU data-cache geometry detection.
//!
//! Cache blocking (panel-block sizing in `mmm`, im2col lowering thresholds, …)
//! is only correct when the block budget is derived from the *actual* cache the
//! code runs on, not a hard-coded constant. This module centralises that
//! detection so every heuristic reads the same memoised numbers instead of each
//! re-implementing a platform probe.
//!
//! All sizes are **bytes**, with `0` meaning "could not detect on this platform"
//! — callers must treat `0` as unknown and fall back conservatively (never
//! over-block a cache you cannot see). The raw fields stay honest; the
//! `*_or_default` helpers apply an architecture-based guess for callers that
//! prefer a number to a zero.
//!
//! Detection is done once, lazily, and cached for the process lifetime.

use std::sync::OnceLock;

/// Detected data-cache sizes in bytes. `0` == unknown on this platform.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheInfo {
    /// L1 data cache (per core), bytes. `0` if unknown.
    pub l1_data: usize,
    /// L2 cache (per perf-core / cluster), bytes. `0` if unknown.
    pub l2: usize,
    /// L3 / last-level cache, bytes. `0` if unknown.
    pub l3: usize,
}

impl CacheInfo {
    /// L1 data cache, or an architecture-based guess when undetected
    /// (64 KiB on arm64, 32 KiB elsewhere — matches common silicon).
    pub fn l1_data_or_default(&self) -> usize {
        if self.l1_data > 0 {
            self.l1_data
        } else if cfg!(target_arch = "aarch64") {
            64 * 1024
        } else {
            32 * 1024
        }
    }

    /// L2 cache, or a conservative 256 KiB guess when undetected.
    pub fn l2_or_default(&self) -> usize {
        if self.l2 > 0 { self.l2 } else { 256 * 1024 }
    }
}

/// Memoised cache geometry for the current machine. Detected once on first call.
pub fn cache_info() -> CacheInfo {
    static CACHE: OnceLock<CacheInfo> = OnceLock::new();
    *CACHE.get_or_init(detect)
}

/// Parse a Linux `/sys` cache `size` string (e.g. `"256K"`, `"8M"`, `"512"`).
#[cfg_attr(not(any(target_os = "linux", target_os = "android")), allow(dead_code))]
fn parse_cache_size(s: &str) -> usize {
    let s = s.trim();
    let (num, mult) = if let Some(n) = s.strip_suffix(['K', 'k']) {
        (n, 1024)
    } else if let Some(n) = s.strip_suffix(['M', 'm']) {
        (n, 1024 * 1024)
    } else {
        (s, 1)
    };
    num.trim().parse::<usize>().unwrap_or(0) * mult
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn detect() -> CacheInfo {
    // Read a scalar `hw.*` sysctl by name via the libc FFI (no subprocess).
    // macOS returns these as a little-endian integer (4 or 8 bytes); a zeroed
    // 8-byte buffer reads either width correctly on little-endian Apple silicon
    // and Intel.
    fn sysctl_usize(name: &str) -> Option<usize> {
        use std::ffi::CString;
        use std::os::raw::{c_char, c_int, c_void};
        unsafe extern "C" {
            fn sysctlbyname(
                name: *const c_char,
                oldp: *mut c_void,
                oldlenp: *mut usize,
                newp: *mut c_void,
                newlen: usize,
            ) -> c_int;
        }
        let cname = CString::new(name).ok()?;
        let mut val: u64 = 0;
        let mut len = std::mem::size_of::<u64>();
        let rc = unsafe {
            sysctlbyname(
                cname.as_ptr(),
                &mut val as *mut u64 as *mut c_void,
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        if rc != 0 || val == 0 { None } else { Some(val as usize) }
    }

    CacheInfo {
        // perflevel0 is the performance cluster on hybrid Apple Silicon.
        l1_data: sysctl_usize("hw.perflevel0.l1dcachesize")
            .or_else(|| sysctl_usize("hw.l1dcachesize"))
            .unwrap_or(0),
        l2: sysctl_usize("hw.perflevel0.l2cachesize")
            .or_else(|| sysctl_usize("hw.l2cachesize"))
            .unwrap_or(0),
        l3: sysctl_usize("hw.perflevel0.l3cachesize")
            .or_else(|| sysctl_usize("hw.l3cachesize"))
            .unwrap_or(0),
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn detect() -> CacheInfo {
    // Walk /sys/.../cache/indexN, keying off the reported level+type rather than
    // assuming a fixed index layout (it varies: SMT, unified vs split L2, …).
    let mut ci = CacheInfo::default();
    for idx in 0..16 {
        let base = format!("/sys/devices/system/cpu/cpu0/cache/index{idx}/");
        let Ok(level) = std::fs::read_to_string(format!("{base}level")) else {
            continue;
        };
        let level: usize = level.trim().parse().unwrap_or(0);
        let ctype = std::fs::read_to_string(format!("{base}type"))
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase();
        let size = std::fs::read_to_string(format!("{base}size"))
            .map(|s| parse_cache_size(&s))
            .unwrap_or(0);
        if size == 0 {
            continue;
        }
        match level {
            1 if ctype == "data" || ctype == "unified" => {
                if ci.l1_data == 0 {
                    ci.l1_data = size;
                }
            }
            2 if ci.l2 == 0 => ci.l2 = size,
            3 if ci.l3 == 0 => ci.l3 = size,
            _ => {}
        }
    }
    ci
}

#[cfg(target_os = "windows")]
fn detect() -> CacheInfo {
    // wmic only reports L2/L3 (in KiB) and is deprecated on Win11; it is the
    // dependency-free option. L1 is left unknown (→ l1_data_or_default).
    // A future GetLogicalProcessorInformationEx probe would also yield L1.
    let mut ci = CacheInfo::default();
    if let Ok(out) = std::process::Command::new("wmic")
        .args(["cpu", "get", "L2CacheSize,L3CacheSize", "/format:value"])
        .output()
    {
        for line in String::from_utf8_lossy(&out.stdout).lines() {
            let line = line.trim();
            if let Some(v) = line.strip_prefix("L2CacheSize=") {
                if let Ok(kb) = v.trim().parse::<usize>() {
                    ci.l2 = kb * 1024;
                }
            } else if let Some(v) = line.strip_prefix("L3CacheSize=") {
                if let Ok(kb) = v.trim().parse::<usize>() {
                    ci.l3 = kb * 1024;
                }
            }
        }
    }
    ci
}

#[cfg(not(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "linux",
    target_os = "android",
    target_os = "windows"
)))]
fn detect() -> CacheInfo {
    // WASM, BSDs, etc.: no portable probe — report unknown, callers fall back.
    CacheInfo::default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_cache_size_units() {
        assert_eq!(parse_cache_size("512"), 512);
        assert_eq!(parse_cache_size("256K"), 256 * 1024);
        assert_eq!(parse_cache_size("8M"), 8 * 1024 * 1024);
        assert_eq!(parse_cache_size(" 1024k "), 1024 * 1024);
        assert_eq!(parse_cache_size("garbage"), 0);
    }

    #[test]
    fn defaults_are_nonzero() {
        let unknown = CacheInfo::default();
        assert!(unknown.l1_data_or_default() >= 32 * 1024);
        assert_eq!(unknown.l2_or_default(), 256 * 1024);
    }

    #[test]
    fn detected_values_are_sane_when_present() {
        // Detection must never panic and must be self-consistent: any level it
        // *does* report should be a plausible power-of-two-ish cache size, and
        // L1 <= L2 <= L3 when all are known.
        let ci = cache_info();
        for (name, v) in [("l1d", ci.l1_data), ("l2", ci.l2), ("l3", ci.l3)] {
            assert!(v == 0 || (1024..=512 * 1024 * 1024).contains(&v), "{name} implausible: {v}");
        }
        if ci.l1_data > 0 && ci.l2 > 0 {
            assert!(ci.l1_data <= ci.l2, "L1 {} > L2 {}", ci.l1_data, ci.l2);
        }
        if ci.l2 > 0 && ci.l3 > 0 {
            assert!(ci.l2 <= ci.l3, "L2 {} > L3 {}", ci.l2, ci.l3);
        }
    }
}
