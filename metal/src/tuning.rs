//! Device performance-tuning profile for the Metal runtime.
//!
//! Every Metal perf constant that used to be scattered as a hardcoded const
//! plus an env escape hatch lives here, resolved ONCE per process into a
//! single [`MetalTuning`] value. Resolution order (lowest to highest
//! precedence):
//!
//! 1. [`MetalTuning::BASELINE`]: the hardcoded defaults. All of them were
//!    tuned on an Apple M4 Pro, 48 GB, during the 2026-08 decode-perf
//!    campaign (models: gpt-oss-20b q40 MoE, qwen3.5-35B-A3B q40 hybrid).
//! 2. Device-informed derivations. None exist today: no baseline value has a
//!    principled scaling rule across device classes yet. When one does (or a
//!    micro-autotuner lands), it belongs in [`MetalTuning::resolve`], between
//!    the baseline and the overrides (e.g. a future
//!    `MetalTuning::from_autotune_cache()` feeding `AppOverrides`).
//! 3. Application overrides ([`set_tuning_overrides`]): programmatic hints an
//!    embedding application registers before the first Metal dispatch.
//! 4. Env overrides: the historical variable names, unchanged semantics,
//!    still highest precedence.
//!
//! Debug/disable escape hatches (`TRACT_METAL_DISABLE_*`, `TRACT_METAL_LOG_*`,
//! `TRACT_METAL_PROFILE_KERNELS`, `TRACT_METAL_GEMM_IMPL`, ...) are NOT part
//! of the profile: they select code paths for debugging, they are not tuned
//! quantities, and some are toggled mid-process by tests. Kernel-coupled
//! constants (e.g. `GDN_CHUNK` = 64, `GDN_COL_BLOCK` = 16 in
//! `gdn_recurrent.rs`) are also excluded: they must match `constant`
//! declarations compiled into the shaders and cannot be tuned host-side.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

use tract_core::internal::*;

/// The resolved per-process Metal performance-tuning profile.
///
/// Read it through [`tuning`]; construct it directly only in tests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalTuning {
    /// Context length (in cached tokens) where decode-shaped SDPA steps
    /// switch from the batched-gemm pipeline to the fused flash-attention
    /// kernel. Benchmarked 2026-08-05 on gpt-oss-20b (M4 Pro): the batched
    /// GGML mm/gemv pipeline beat the flash kernel at every length tried
    /// (74, 2800, 5.6k, 11k ctx: e.g. 24.3 vs 21.1 tok/s at 11k), so flash
    /// is disabled by default (`usize::MAX`) and kept for future tuning.
    /// Env: `TRACT_METAL_FLASH_SDPA_MIN_T` (legacy
    /// `TRACT_METAL_GPT_OSS_FLASH_MIN_T`).
    pub flash_sdpa_min_t: usize,
    /// Key-block size of the block-wise prefill attention: prefill steps
    /// whose effective key length exceeds this loop key-blocks through a
    /// fixed scores buffer with a running-max/denominator f32 state instead
    /// of materializing the full [Hkv, group*S, T] scores/probs (which near
    /// 50k context reaches GBs per layer, touched 3-4 times). 0 disables.
    /// Env: `TRACT_METAL_SDPA_PREFILL_BLOCK`.
    pub sdpa_prefill_block: usize,
    /// Context length where the decode AV gemv switches to split-k (partial
    /// sums over key chunks + a small reduce). Measured crossover vs the
    /// plain batched gemv on gpt-oss-20b (M4 Pro): plain wins to ~5.6k
    /// (52.6 vs 47.5 tok/s @2800), split-k wins beyond (39.2 vs 37.6 @11k).
    /// No env override (gate: `TRACT_METAL_DISABLE_SDPA_SPLIT_K`).
    pub sdpa_split_k_min_t: usize,
    /// Target keys per split-k chunk (chunk count is clamped to [2, 16]).
    /// No env override.
    pub sdpa_split_k_chunk: usize,
    /// Commit cadence: split each forward into command buffers every N
    /// `command_buffer()` acquisitions (roughly every N kernel dispatches),
    /// so the GPU starts executing early layers while the CPU still encodes
    /// late ones. 0 (the default) disables the cadence: single buffer per
    /// forward, plus whatever boundaries ops request themselves. The right
    /// value is model-shaped, not device-shaped: hybrid-attention models
    /// (~1200 small dispatches/token, qwen3.5-35B) want 10 (interleaved A/B
    /// vs 20: +1.0 tok/s @74, +1.2 @2799, +1.8 @11k; 4-6 too eager, 12
    /// equal, >=16 loses), while dense-KV models (gpt-oss-20b, few large
    /// dispatches) lose badly under any cadence (66 -> 53 tok/s), hence
    /// baseline 0 and a per-model application hint
    /// ([`set_tuning_overrides`]). Env:
    /// `TRACT_METAL_COMMIT_EVERY_N_DISPATCHES`.
    pub commit_every_n_dispatches: usize,
    /// How many committed-but-unawaited command buffers `commit_current`
    /// keeps in flight (>= 1). The wait on the oldest buffer is the
    /// backpressure that bounds transient memory (without it, a long-context
    /// forward retains every layer's transients at once and thrashes).
    /// Depth 2 already overlaps CPU encoding with GPU execution, but any
    /// encode hiccup then drains the queue and the GPU idles between buffers
    /// (~2.8 ms/step measured on qwen3.5-35B decode at cadence 10, ~115
    /// buffers/step). Depth 8 absorbs the jitter (+5 tok/s on that model,
    /// +4 at 11k ctx) while transients stay bounded: post-arena they are
    /// almost all views into the session arena, so deeper retention holds
    /// Arc clones, not extra wired memory. Env: `TRACT_METAL_MAX_IN_FLIGHT`.
    pub max_command_buffers_in_flight: usize,
    /// Routed-MoE ops end their dispatch with a non-blocking command-buffer
    /// boundary only when the step routes more than this many (token,
    /// expert) pairs. The boundary is a correctness measure for prefill-sized
    /// batches (see `MetalRoutedQ40MatMul`); decode steps (route_count =
    /// top_k) end with the runtime's own blocking logits sync anyway, and
    /// splitting at decode costs ~75% of the decode wall time in
    /// waitUntilCompleted for no benefit. Env:
    /// `TRACT_METAL_MOE_COMMIT_MIN_ROUTES`.
    pub moe_commit_min_routes: usize,
    /// Route count at which routed-q40 MoE matmuls take the expert-grouped
    /// path: bin the routes by expert (single-threadgroup counting sort),
    /// then let each threadgroup amortize every weight read across 32 routes
    /// of one expert through the simdgroup-matrix pipeline. Halves
    /// 2800-token prefill vs the per-route gemv (11.5 -> 5.9 s on
    /// gpt-oss-20b). The per-route kernel re-reads an expert's weights once
    /// per route, so below this threshold (decode-sized route lists) it is
    /// the cheaper one. No env override (gate:
    /// `TRACT_METAL_DISABLE_GROUPED_MOE`).
    pub moe_grouped_min_routes: usize,
    /// Hard cap in bytes on recycled buffer-pool memory. The budget must
    /// hold the session memory arena (recycled once per decode step, tens to
    /// hundreds of MB at long context) with room to spare for the small
    /// fixed-shape transients; entries beyond it are evicted oldest-first,
    /// so stale shapes from a grown context cannot pin wired memory forever
    /// (unbounded pinning is what used to slow the weight-streaming kernels
    /// when large buffers were pooled without eviction). Env:
    /// `TRACT_METAL_POOL_MAX_MB` (in MiB).
    pub pool_max_bytes: usize,
    /// Cap on recycled buffers pooled per exact (dtype, shape) key.
    /// No env override.
    pub max_pooled_per_key: usize,
}

/// Partial profile an embedding application may register through
/// [`set_tuning_overrides`] before the first Metal dispatch. `None` fields
/// keep the baseline; env vars still win over these hints.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MetalTuningOverrides {
    pub flash_sdpa_min_t: Option<usize>,
    pub sdpa_prefill_block: Option<usize>,
    pub sdpa_split_k_min_t: Option<usize>,
    pub sdpa_split_k_chunk: Option<usize>,
    pub commit_every_n_dispatches: Option<usize>,
    pub max_command_buffers_in_flight: Option<usize>,
    pub moe_commit_min_routes: Option<usize>,
    pub moe_grouped_min_routes: Option<usize>,
    pub pool_max_bytes: Option<usize>,
    pub max_pooled_per_key: Option<usize>,
}

impl MetalTuning {
    /// The hardcoded defaults, tuned on an Apple M4 Pro (48 GB) during the
    /// 2026-08 decode-perf campaign. See each field for provenance.
    pub const BASELINE: MetalTuning = MetalTuning {
        flash_sdpa_min_t: usize::MAX,
        sdpa_prefill_block: 4096,
        sdpa_split_k_min_t: 8192,
        sdpa_split_k_chunk: 2048,
        commit_every_n_dispatches: 0,
        max_command_buffers_in_flight: 8,
        moe_commit_min_routes: 64,
        moe_grouped_min_routes: 64,
        pool_max_bytes: 512 * 1024 * 1024,
        max_pooled_per_key: 16,
    };

    /// Resolve the profile from the baseline, the registered application
    /// overrides and the process environment. Called once per process by
    /// [`tuning`]; use that accessor instead.
    pub fn from_env_and_device() -> MetalTuning {
        let overrides = app_overrides().lock().map(|o| o.clone()).unwrap_or_default();
        Self::resolve(&overrides, |name| std::env::var(name).ok())
    }

    /// Pure resolution, env injected for hermetic tests.
    fn resolve(
        overrides: &MetalTuningOverrides,
        env: impl Fn(&str) -> Option<String>,
    ) -> MetalTuning {
        let base = MetalTuning::BASELINE;
        // Seam for device-informed derivations and a future
        // `from_autotune_cache()`: adjust `base` here, between the baseline
        // and the overrides. Nothing today has a principled cross-device
        // scaling rule, so `base` is used as-is.
        let parsed = |name: &str| env(name).and_then(|v| v.parse::<usize>().ok());
        MetalTuning {
            flash_sdpa_min_t: parsed("TRACT_METAL_FLASH_SDPA_MIN_T")
                .or_else(|| parsed("TRACT_METAL_GPT_OSS_FLASH_MIN_T"))
                .or(overrides.flash_sdpa_min_t)
                .unwrap_or(base.flash_sdpa_min_t),
            sdpa_prefill_block: parsed("TRACT_METAL_SDPA_PREFILL_BLOCK")
                .or(overrides.sdpa_prefill_block)
                .unwrap_or(base.sdpa_prefill_block),
            sdpa_split_k_min_t: overrides.sdpa_split_k_min_t.unwrap_or(base.sdpa_split_k_min_t),
            sdpa_split_k_chunk: overrides.sdpa_split_k_chunk.unwrap_or(base.sdpa_split_k_chunk),
            commit_every_n_dispatches: parsed("TRACT_METAL_COMMIT_EVERY_N_DISPATCHES")
                .or(overrides.commit_every_n_dispatches)
                .unwrap_or(base.commit_every_n_dispatches),
            max_command_buffers_in_flight: parsed("TRACT_METAL_MAX_IN_FLIGHT")
                .filter(|&n| n >= 1)
                .or(overrides.max_command_buffers_in_flight.filter(|&n| n >= 1))
                .unwrap_or(base.max_command_buffers_in_flight),
            moe_commit_min_routes: parsed("TRACT_METAL_MOE_COMMIT_MIN_ROUTES")
                .or(overrides.moe_commit_min_routes)
                .unwrap_or(base.moe_commit_min_routes),
            moe_grouped_min_routes: overrides
                .moe_grouped_min_routes
                .unwrap_or(base.moe_grouped_min_routes),
            pool_max_bytes: parsed("TRACT_METAL_POOL_MAX_MB")
                .map(|mb| mb * 1024 * 1024)
                .or(overrides.pool_max_bytes)
                .unwrap_or(base.pool_max_bytes),
            max_pooled_per_key: overrides.max_pooled_per_key.unwrap_or(base.max_pooled_per_key),
        }
    }
}

fn app_overrides() -> &'static Mutex<MetalTuningOverrides> {
    static O: OnceLock<Mutex<MetalTuningOverrides>> = OnceLock::new();
    O.get_or_init(|| Mutex::new(MetalTuningOverrides::default()))
}

static RESOLVED: AtomicBool = AtomicBool::new(false);

/// The process-wide resolved tuning profile. First call resolves (and logs)
/// it; later calls are a static read. Env lookups on per-dispatch hot paths
/// measurably show up in decode CPU profiles, hence resolve-once.
pub fn tuning() -> &'static MetalTuning {
    static T: OnceLock<MetalTuning> = OnceLock::new();
    T.get_or_init(|| {
        RESOLVED.store(true, Ordering::Release);
        let tuning = MetalTuning::from_env_and_device();
        log::debug!("resolved Metal tuning profile: {tuning:?}");
        tuning
    })
}

/// Register application tuning hints. Must be called before the first Metal
/// dispatch of the process (i.e. before runtime prepare); fails once the
/// profile has been resolved. Later calls merge over earlier ones field by
/// field. Env vars still take precedence over these hints.
pub fn set_tuning_overrides(overrides: MetalTuningOverrides) -> TractResult<()> {
    let mut current = app_overrides().lock().map_err(|e| anyhow!("{e}"))?;
    if RESOLVED.load(Ordering::Acquire) {
        bail!(
            "Metal tuning profile already resolved; \
             set_tuning_overrides must run before the first Metal dispatch"
        );
    }
    macro_rules! merge {
        ($($field:ident),*) => {
            $(if let Some(v) = overrides.$field { current.$field = Some(v); })*
        };
    }
    merge!(
        flash_sdpa_min_t,
        sdpa_prefill_block,
        sdpa_split_k_min_t,
        sdpa_split_k_chunk,
        commit_every_n_dispatches,
        max_command_buffers_in_flight,
        moe_commit_min_routes,
        moe_grouped_min_routes,
        pool_max_bytes,
        max_pooled_per_key
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_env(_: &str) -> Option<String> {
        None
    }

    /// The documented baseline IS the default resolution: no env, no
    /// overrides must yield exactly the 2026-08 M4 Pro tuned values.
    #[test]
    fn default_profile_is_baseline() {
        let t = MetalTuning::resolve(&MetalTuningOverrides::default(), no_env);
        assert_eq!(t, MetalTuning::BASELINE);
        assert_eq!(t.flash_sdpa_min_t, usize::MAX);
        assert_eq!(t.sdpa_prefill_block, 4096);
        assert_eq!(t.sdpa_split_k_min_t, 8192);
        assert_eq!(t.sdpa_split_k_chunk, 2048);
        assert_eq!(t.commit_every_n_dispatches, 0);
        assert_eq!(t.max_command_buffers_in_flight, 8);
        assert_eq!(t.moe_commit_min_routes, 64);
        assert_eq!(t.moe_grouped_min_routes, 64);
        assert_eq!(t.pool_max_bytes, 512 * 1024 * 1024);
        assert_eq!(t.max_pooled_per_key, 16);
    }

    #[test]
    fn env_override_wins_over_baseline_and_hints() {
        let env = |name: &str| match name {
            "TRACT_METAL_COMMIT_EVERY_N_DISPATCHES" => Some("20".to_string()),
            "TRACT_METAL_MAX_IN_FLIGHT" => Some("2".to_string()),
            "TRACT_METAL_POOL_MAX_MB" => Some("128".to_string()),
            "TRACT_METAL_SDPA_PREFILL_BLOCK" => Some("1024".to_string()),
            "TRACT_METAL_MOE_COMMIT_MIN_ROUTES" => Some("32".to_string()),
            _ => None,
        };
        let hints = MetalTuningOverrides {
            commit_every_n_dispatches: Some(10),
            ..Default::default()
        };
        let t = MetalTuning::resolve(&hints, env);
        // Env beats the application hint.
        assert_eq!(t.commit_every_n_dispatches, 20);
        assert_eq!(t.max_command_buffers_in_flight, 2);
        assert_eq!(t.pool_max_bytes, 128 * 1024 * 1024);
        assert_eq!(t.sdpa_prefill_block, 1024);
        assert_eq!(t.moe_commit_min_routes, 32);
        // Untouched fields keep the baseline.
        assert_eq!(t.flash_sdpa_min_t, usize::MAX);
        assert_eq!(t.moe_grouped_min_routes, 64);
    }

    #[test]
    fn app_hint_wins_over_baseline() {
        let hints = MetalTuningOverrides {
            commit_every_n_dispatches: Some(10),
            ..Default::default()
        };
        let t = MetalTuning::resolve(&hints, no_env);
        assert_eq!(t.commit_every_n_dispatches, 10);
        assert_eq!(t.max_command_buffers_in_flight, 8);
    }

    /// Legacy GPT-OSS-era env name still works, new name wins over it.
    #[test]
    fn flash_min_t_legacy_env_fallback() {
        let legacy_only = |name: &str| {
            (name == "TRACT_METAL_GPT_OSS_FLASH_MIN_T").then(|| "4096".to_string())
        };
        let t = MetalTuning::resolve(&MetalTuningOverrides::default(), legacy_only);
        assert_eq!(t.flash_sdpa_min_t, 4096);
        let both = |name: &str| match name {
            "TRACT_METAL_FLASH_SDPA_MIN_T" => Some("0".to_string()),
            "TRACT_METAL_GPT_OSS_FLASH_MIN_T" => Some("4096".to_string()),
            _ => None,
        };
        let t = MetalTuning::resolve(&MetalTuningOverrides::default(), both);
        assert_eq!(t.flash_sdpa_min_t, 0);
    }

    /// Unparseable or out-of-domain env values fall through, matching the
    /// historical per-site parsing (`parse().ok()`, in-flight `>= 1`).
    #[test]
    fn invalid_env_values_fall_through() {
        let env = |name: &str| match name {
            "TRACT_METAL_MAX_IN_FLIGHT" => Some("0".to_string()),
            "TRACT_METAL_COMMIT_EVERY_N_DISPATCHES" => Some("not-a-number".to_string()),
            _ => None,
        };
        let t = MetalTuning::resolve(&MetalTuningOverrides::default(), env);
        assert_eq!(t.max_command_buffers_in_flight, 8);
        assert_eq!(t.commit_every_n_dispatches, 0);
    }
}
