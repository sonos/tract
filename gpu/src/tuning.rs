//! Backend-agnostic GPU tuning profile.
//!
//! Same resolve-once design as `tract_metal::tuning` (see that module for
//! the rationale, the seam for device-informed derivations and the offline
//! autotune cache): baseline defaults, then env overrides, resolved once per
//! process and logged at debug level. Backend-specific constants (command
//! buffer cadence, pool caps, kernel thresholds) live in the backend crates;
//! only values meaningful to every GPU runtime belong here.

use std::sync::OnceLock;

/// The resolved per-process GPU tuning profile. Read it through [`tuning`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuTuning {
    /// Representative value for symbolic dimensions missing from the memory
    /// schema sizing hint (typically the sequence/context length). It only
    /// drives the arena partition packing order, never correctness: an
    /// unrepresentative value yields a valid but less optimally packed
    /// schema. Default 1024, in use unchanged since the device memory arena
    /// landed (2026-08 campaign, M4 Pro 48GB). Env:
    /// `TRACT_GPU_MEM_HINT_DEFAULT`.
    pub mem_hint_default_dim: i64,
}

impl GpuTuning {
    /// The hardcoded defaults. See each field for provenance.
    pub const BASELINE: GpuTuning = GpuTuning { mem_hint_default_dim: 1024 };

    /// Resolve from baseline + env. Called once per process by [`tuning`].
    pub fn from_env_and_device() -> GpuTuning {
        Self::resolve(|name| std::env::var(name).ok())
    }

    /// Pure resolution, env injected for hermetic tests.
    fn resolve(env: impl Fn(&str) -> Option<String>) -> GpuTuning {
        let base = GpuTuning::BASELINE;
        GpuTuning {
            mem_hint_default_dim: env("TRACT_GPU_MEM_HINT_DEFAULT")
                .and_then(|v| v.parse().ok())
                .unwrap_or(base.mem_hint_default_dim),
        }
    }
}

/// The process-wide resolved GPU tuning profile.
pub fn tuning() -> &'static GpuTuning {
    static T: OnceLock<GpuTuning> = OnceLock::new();
    T.get_or_init(|| {
        let tuning = GpuTuning::from_env_and_device();
        log::debug!("resolved GPU tuning profile: {tuning:?}");
        tuning
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_profile_is_baseline() {
        let t = GpuTuning::resolve(|_| None);
        assert_eq!(t, GpuTuning::BASELINE);
        assert_eq!(t.mem_hint_default_dim, 1024);
    }

    #[test]
    fn env_override_wins() {
        let t = GpuTuning::resolve(|name| {
            (name == "TRACT_GPU_MEM_HINT_DEFAULT").then(|| "4096".to_string())
        });
        assert_eq!(t.mem_hint_default_dim, 4096);
    }
}
