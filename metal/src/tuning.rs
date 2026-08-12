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
//!    principled scaling rule across device classes yet. When one does, it
//!    belongs in [`MetalTuning::resolve`], between the baseline and the
//!    autotune cache.
//! 3. Autotune cache ([`AutotuneCache`]): a JSON file written by an OFFLINE
//!    sweep tool (e.g. ohana's `tune_decode` example), one file per device.
//!    Default location `~/.cache/tract/tuning/<sanitized-device-name>.json`
//!    (see [`sanitize_device_name`]); `TRACT_METAL_TUNING_CACHE` overrides
//!    the location and may name either a directory (the per-device file is
//!    looked up inside it) or a file. Within the cache, the `device` section
//!    applies first, then the model section selected via
//!    [`set_tuning_model_key`] (if any). `TRACT_METAL_DISABLE_TUNING_CACHE=1`
//!    skips reading it. A malformed cache file is never a session error: it
//!    is logged (warn) and ignored, as are unknown fields (forward compat).
//! 4. Application overrides ([`set_tuning_overrides`]): programmatic hints an
//!    embedding application registers before the first Metal dispatch.
//! 5. Env overrides: the historical variable names, unchanged semantics,
//!    still highest precedence.
//!
//! # Offline autotuning workflow
//!
//! Sessions never probe at load time: tuning cost is paid once, offline.
//! Run ohana's `tune_decode --write` once per device (optionally once per
//! model class with `--model-key <key>`); it sweeps the output-invariant
//! scheduling knobs against a real model and writes the cache file above.
//! Every later tract session on that device picks the cache up
//! automatically; apps opt a model into its section with
//! [`set_tuning_model_key`]; env vars still win for one-off experiments.
//!
//! Debug/disable escape hatches (`TRACT_METAL_DISABLE_*`, `TRACT_METAL_LOG_*`,
//! `TRACT_METAL_PROFILE_KERNELS`, `TRACT_METAL_GEMM_IMPL`, ...) are NOT part
//! of the profile: they select code paths for debugging, they are not tuned
//! quantities, and some are toggled mid-process by tests. Kernel-coupled
//! constants (e.g. `GDN_CHUNK` = 64, `GDN_COL_BLOCK` = 16 in
//! `gdn_recurrent.rs`) are also excluded: they must match `constant`
//! declarations compiled into the shaders and cannot be tuned host-side.

use std::path::{Path, PathBuf};
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

/// Names every settable field once, so the field list cannot drift between
/// the merge logic, the cache parser and the debug formatting.
macro_rules! for_each_tuning_field {
    ($m:ident) => {
        $m!(
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
        )
    };
}

impl MetalTuningOverrides {
    /// Merge `other` over `self`, field by field: `Some` fields of `other`
    /// win, `None` fields keep the current value.
    fn merge_from(&mut self, other: &MetalTuningOverrides) {
        macro_rules! merge {
            ($($field:ident),*) => {
                $(if let Some(v) = other.$field { self.$field = Some(v); })*
            };
        }
        for_each_tuning_field!(merge);
    }

    /// `field=value` list of the set fields, for the applied-cache debug log.
    fn set_fields(&self) -> String {
        let mut out = Vec::new();
        macro_rules! fmt {
            ($($field:ident),*) => {
                $(if let Some(v) = self.$field {
                    out.push(format!(concat!(stringify!($field), "={}"), v));
                })*
            };
        }
        for_each_tuning_field!(fmt);
        if out.is_empty() { "<none>".to_string() } else { out.join(" ") }
    }

    /// Set a field by its cache-file name. `Err` on an unknown name.
    fn set_by_name(&mut self, name: &str, value: usize) -> TractResult<()> {
        macro_rules! set {
            ($($field:ident),*) => {
                match name {
                    $(stringify!($field) => { self.$field = Some(value); Ok(()) })*
                    _ => bail!("unknown tuning field `{name}`"),
                }
            };
        }
        for_each_tuning_field!(set)
    }
}

/// The parsed autotune cache file: tuned values measured offline by a sweep
/// tool (see the module docs for the workflow and the resolution order).
///
/// File format (JSON, one file per device):
///
/// ```json
/// {
///   "schema_version": 1,
///   "device_name": "Apple M4 Pro",
///   "written_by": "tune_decode (ohana)",
///   "date": "2026-08-12T10:00:00Z",
///   "device": { "max_command_buffers_in_flight": 8 },
///   "models": { "hybrid-gdn": { "commit_every_n_dispatches": 10 } }
/// }
/// ```
///
/// The `device` section applies to every session on the device; a `models`
/// section applies on top of it when the application selects its key with
/// [`set_tuning_model_key`]. Field names are the [`MetalTuning`] field names,
/// values are non-negative integers (`pool_max_bytes` in bytes). Unknown
/// fields warn and are ignored (a newer tool may know fields this build does
/// not); a malformed file warns and is ignored entirely.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AutotuneCache {
    /// Where the cache was read from (for logs).
    pub path: PathBuf,
    /// Device-wide tuned values.
    pub device: MetalTuningOverrides,
    /// Model-keyed tuned values, applied over `device` when selected.
    pub models: Vec<(String, MetalTuningOverrides)>,
}

/// The autotune cache schema version this build reads and writes.
pub const AUTOTUNE_CACHE_SCHEMA_VERSION: u64 = 1;

impl AutotuneCache {
    /// Read and parse a cache file. Any problem (unreadable, malformed,
    /// wrong schema version) is logged at warn level and yields `None`:
    /// a bad cache must never break a session.
    pub fn load(path: &Path) -> Option<AutotuneCache> {
        let text = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(e) => {
                log::warn!("ignoring unreadable Metal tuning cache {}: {e}", path.display());
                return None;
            }
        };
        match Self::parse(path, &text) {
            Ok(cache) => Some(cache),
            Err(e) => {
                log::warn!("ignoring malformed Metal tuning cache {}: {e:?}", path.display());
                None
            }
        }
    }

    fn parse(path: &Path, text: &str) -> TractResult<AutotuneCache> {
        let root: serde_json::Value = serde_json::from_str(text)?;
        let root = root.as_object().context("cache root must be a JSON object")?;
        let version = root
            .get("schema_version")
            .and_then(|v| v.as_u64())
            .context("missing or non-integer schema_version")?;
        if version != AUTOTUNE_CACHE_SCHEMA_VERSION {
            bail!("unsupported schema_version {version} (this build reads {AUTOTUNE_CACHE_SCHEMA_VERSION})");
        }
        let mut cache = AutotuneCache { path: path.to_path_buf(), ..Default::default() };
        for (key, value) in root {
            match key.as_str() {
                // Provenance metadata: informational only.
                "schema_version" | "device_name" | "written_by" | "date" => (),
                "device" => {
                    let section =
                        value.as_object().context("`device` section must be a JSON object")?;
                    cache.device = Self::parse_section(path, "device", section);
                }
                "models" => {
                    let models =
                        value.as_object().context("`models` section must be a JSON object")?;
                    for (model_key, section) in models {
                        let section = section.as_object().with_context(|| {
                            format!("models.{model_key} section must be a JSON object")
                        })?;
                        let overrides =
                            Self::parse_section(path, &format!("models.{model_key}"), section);
                        cache.models.push((model_key.clone(), overrides));
                    }
                }
                unknown => log::warn!(
                    "Metal tuning cache {}: ignoring unknown field `{unknown}`",
                    path.display()
                ),
            }
        }
        Ok(cache)
    }

    /// Parse one field->value section. Unknown fields and out-of-domain
    /// values warn and are skipped, never an error (forward compat).
    fn parse_section(
        path: &Path,
        section: &str,
        obj: &serde_json::Map<String, serde_json::Value>,
    ) -> MetalTuningOverrides {
        let mut overrides = MetalTuningOverrides::default();
        for (name, value) in obj {
            let Some(value) = value.as_u64().and_then(|v| usize::try_from(v).ok()) else {
                log::warn!(
                    "Metal tuning cache {} [{section}]: ignoring `{name}`: \
                     value must be a non-negative integer",
                    path.display()
                );
                continue;
            };
            if name == "max_command_buffers_in_flight" && value < 1 {
                log::warn!(
                    "Metal tuning cache {} [{section}]: ignoring \
                     max_command_buffers_in_flight=0 (must be >= 1)",
                    path.display()
                );
                continue;
            }
            if let Err(e) = overrides.set_by_name(name, value) {
                log::warn!("Metal tuning cache {} [{section}]: ignoring `{name}`: {e}", path.display());
            }
        }
        overrides
    }

    /// The model section for `key`, if present.
    fn model_section(&self, key: &str) -> Option<&MetalTuningOverrides> {
        self.models.iter().find(|(k, _)| k == key).map(|(_, s)| s)
    }
}

/// `Metal device name -> cache file stem`: lowercased, every non-alphanumeric
/// run collapsed to a single `-` (e.g. "Apple M4 Pro" -> "apple-m4-pro").
pub fn sanitize_device_name(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for c in name.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else if !out.ends_with('-') {
            out.push('-');
        }
    }
    out.trim_matches('-').to_string()
}

/// The name of the system default Metal device, if one exists.
pub fn current_device_name() -> Option<String> {
    metal::Device::system_default().map(|d| d.name().to_string())
}

/// The default autotune cache path for a device:
/// `~/.cache/tract/tuning/<sanitized-device-name>.json`. `None` when `HOME`
/// is unset or the device name sanitizes to nothing.
pub fn autotune_cache_default_path(device_name: &str) -> Option<PathBuf> {
    let stem = sanitize_device_name(device_name);
    if stem.is_empty() {
        return None;
    }
    let home = std::env::var_os("HOME")?;
    Some(PathBuf::from(home).join(".cache/tract/tuning").join(format!("{stem}.json")))
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

    /// Resolve the profile from the baseline, the autotune cache, the
    /// registered application overrides and the process environment. Called
    /// once per process by [`tuning`]; use that accessor instead.
    pub fn from_env_and_device() -> MetalTuning {
        let overrides = app_overrides().lock().map(|o| o.clone()).unwrap_or_default();
        let model_key = tuning_model_key().lock().map(|k| k.clone()).unwrap_or_default();
        let env = |name: &str| std::env::var(name).ok();
        let cache = Self::from_autotune_cache(&env);
        if let Some(cache) = &cache {
            let model = model_key
                .as_deref()
                .and_then(|key| Some((key, cache.model_section(key)?.set_fields())));
            log::debug!(
                "applying Metal autotune cache {}: device section: {}, model section{}",
                cache.path.display(),
                cache.device.set_fields(),
                match &model {
                    Some((key, fields)) => format!(" [{key}]: {fields}"),
                    None => format!(": <none> (model key: {model_key:?})"),
                }
            );
        }
        Self::resolve(cache.as_ref(), model_key.as_deref(), &overrides, env)
    }

    /// Locate and load the autotune cache, env injected for hermetic tests.
    /// `TRACT_METAL_DISABLE_TUNING_CACHE=1` skips it; `TRACT_METAL_TUNING_CACHE`
    /// overrides the default per-device path and may name a directory (the
    /// `<sanitized-device-name>.json` file is looked up inside it) or a file.
    fn from_autotune_cache(env: &impl Fn(&str) -> Option<String>) -> Option<AutotuneCache> {
        if env("TRACT_METAL_DISABLE_TUNING_CACHE").as_deref() == Some("1") {
            log::debug!("Metal autotune cache disabled by TRACT_METAL_DISABLE_TUNING_CACHE");
            return None;
        }
        let path = match env("TRACT_METAL_TUNING_CACHE").map(PathBuf::from) {
            Some(path) if path.is_dir() => {
                let stem = sanitize_device_name(&current_device_name()?);
                path.join(format!("{stem}.json"))
            }
            Some(path) => path,
            None => autotune_cache_default_path(&current_device_name()?)?,
        };
        if !path.is_file() {
            return None;
        }
        AutotuneCache::load(&path)
    }

    /// Pure resolution, cache and env injected for hermetic tests.
    fn resolve(
        cache: Option<&AutotuneCache>,
        model_key: Option<&str>,
        overrides: &MetalTuningOverrides,
        env: impl Fn(&str) -> Option<String>,
    ) -> MetalTuning {
        let base = MetalTuning::BASELINE;
        // Seam for device-informed derivations: adjust `base` here, between
        // the baseline and the autotune cache. Nothing today has a principled
        // cross-device scaling rule, so `base` is used as-is.
        // Autotune cache and app hints collapse into one override layer:
        // cache device section, then the selected cache model section, then
        // the app overrides, later layers winning field by field.
        let mut merged = MetalTuningOverrides::default();
        if let Some(cache) = cache {
            merged.merge_from(&cache.device);
            if let Some(section) = model_key.and_then(|key| cache.model_section(key)) {
                merged.merge_from(section);
            }
        }
        merged.merge_from(overrides);
        let overrides = &merged;
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

fn tuning_model_key() -> &'static Mutex<Option<String>> {
    static K: OnceLock<Mutex<Option<String>>> = OnceLock::new();
    K.get_or_init(|| Mutex::new(None))
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
    current.merge_from(&overrides);
    Ok(())
}

/// Select the autotune-cache model section this process should apply (e.g.
/// `"hybrid-gdn"`), on top of the cache's device section. A no-op when no
/// cache exists or it has no such section. Same contract as
/// [`set_tuning_overrides`]: must run before the first Metal dispatch of the
/// process, fails once the profile has been resolved; env vars and app
/// overrides still win over the cache.
pub fn set_tuning_model_key(key: impl Into<String>) -> TractResult<()> {
    let mut current = tuning_model_key().lock().map_err(|e| anyhow!("{e}"))?;
    if RESOLVED.load(Ordering::Acquire) {
        bail!(
            "Metal tuning profile already resolved; \
             set_tuning_model_key must run before the first Metal dispatch"
        );
    }
    *current = Some(key.into());
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
        let t = MetalTuning::resolve(None, None, &MetalTuningOverrides::default(), no_env);
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
        let t = MetalTuning::resolve(None, None, &hints, env);
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
        let t = MetalTuning::resolve(None, None, &hints, no_env);
        assert_eq!(t.commit_every_n_dispatches, 10);
        assert_eq!(t.max_command_buffers_in_flight, 8);
    }

    /// Legacy GPT-OSS-era env name still works, new name wins over it.
    #[test]
    fn flash_min_t_legacy_env_fallback() {
        let legacy_only = |name: &str| {
            (name == "TRACT_METAL_GPT_OSS_FLASH_MIN_T").then(|| "4096".to_string())
        };
        let t = MetalTuning::resolve(None, None, &MetalTuningOverrides::default(), legacy_only);
        assert_eq!(t.flash_sdpa_min_t, 4096);
        let both = |name: &str| match name {
            "TRACT_METAL_FLASH_SDPA_MIN_T" => Some("0".to_string()),
            "TRACT_METAL_GPT_OSS_FLASH_MIN_T" => Some("4096".to_string()),
            _ => None,
        };
        let t = MetalTuning::resolve(None, None, &MetalTuningOverrides::default(), both);
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
        let t = MetalTuning::resolve(None, None, &MetalTuningOverrides::default(), env);
        assert_eq!(t.max_command_buffers_in_flight, 8);
        assert_eq!(t.commit_every_n_dispatches, 0);
    }

    fn temp_cache_file(name: &str, content: &str) -> PathBuf {
        let path = std::env::temp_dir()
            .join(format!("tract-metal-tuning-test-{}-{name}.json", std::process::id()));
        std::fs::write(&path, content).unwrap();
        path
    }

    const CACHE_JSON: &str = r#"{
        "schema_version": 1,
        "device_name": "Apple M4 Pro",
        "written_by": "unit test",
        "date": "2026-08-12",
        "device": { "max_command_buffers_in_flight": 4, "moe_commit_min_routes": 128 },
        "models": { "hybrid-gdn": { "commit_every_n_dispatches": 10, "moe_commit_min_routes": 32 } }
    }"#;

    #[test]
    fn cache_device_section_applies_over_baseline() {
        let path = temp_cache_file("device-section", CACHE_JSON);
        let cache = AutotuneCache::load(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        let t = MetalTuning::resolve(Some(&cache), None, &MetalTuningOverrides::default(), no_env);
        assert_eq!(t.max_command_buffers_in_flight, 4);
        assert_eq!(t.moe_commit_min_routes, 128);
        // Model sections are inert without a selected key.
        assert_eq!(t.commit_every_n_dispatches, 0);
        // Untouched fields keep the baseline.
        assert_eq!(t.pool_max_bytes, MetalTuning::BASELINE.pool_max_bytes);
    }

    #[test]
    fn cache_model_section_applies_over_device_section() {
        let path = temp_cache_file("model-section", CACHE_JSON);
        let cache = AutotuneCache::load(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        let t = MetalTuning::resolve(
            Some(&cache),
            Some("hybrid-gdn"),
            &MetalTuningOverrides::default(),
            no_env,
        );
        assert_eq!(t.commit_every_n_dispatches, 10);
        assert_eq!(t.moe_commit_min_routes, 32); // model section beats device section
        assert_eq!(t.max_command_buffers_in_flight, 4); // device section still applies
        // An unknown key falls back to the device section alone.
        let t = MetalTuning::resolve(
            Some(&cache),
            Some("no-such-model"),
            &MetalTuningOverrides::default(),
            no_env,
        );
        assert_eq!(t.commit_every_n_dispatches, 0);
        assert_eq!(t.moe_commit_min_routes, 128);
    }

    #[test]
    fn app_hint_and_env_win_over_cache() {
        let path = temp_cache_file("precedence", CACHE_JSON);
        let cache = AutotuneCache::load(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        let hints =
            MetalTuningOverrides { moe_commit_min_routes: Some(96), ..Default::default() };
        // App hint beats both cache sections.
        let t = MetalTuning::resolve(Some(&cache), Some("hybrid-gdn"), &hints, no_env);
        assert_eq!(t.moe_commit_min_routes, 96);
        // Env beats the app hint and the cache.
        let env =
            |name: &str| (name == "TRACT_METAL_MOE_COMMIT_MIN_ROUTES").then(|| "48".to_string());
        let t = MetalTuning::resolve(Some(&cache), Some("hybrid-gdn"), &hints, env);
        assert_eq!(t.moe_commit_min_routes, 48);
        // Env beats a cache field no hint touches.
        let env = |name: &str| (name == "TRACT_METAL_MAX_IN_FLIGHT").then(|| "16".to_string());
        let t = MetalTuning::resolve(Some(&cache), None, &MetalTuningOverrides::default(), env);
        assert_eq!(t.max_command_buffers_in_flight, 16);
    }

    #[test]
    fn disable_flag_skips_cache() {
        let path = temp_cache_file("disable-flag", CACHE_JSON);
        let path_str = path.to_string_lossy().to_string();
        let with_cache = |name: &str| match name {
            "TRACT_METAL_TUNING_CACHE" => Some(path_str.clone()),
            _ => None,
        };
        assert!(MetalTuning::from_autotune_cache(&with_cache).is_some());
        let disabled = |name: &str| match name {
            "TRACT_METAL_TUNING_CACHE" => Some(path_str.clone()),
            "TRACT_METAL_DISABLE_TUNING_CACHE" => Some("1".to_string()),
            _ => None,
        };
        assert!(MetalTuning::from_autotune_cache(&disabled).is_none());
        std::fs::remove_file(&path).unwrap();
    }

    /// A malformed cache file is warn-and-ignore, never a session error.
    #[test]
    fn malformed_cache_is_ignored() {
        for (name, content) in [
            ("not-json", "{ this is not json"),
            ("not-object", "[1, 2, 3]"),
            ("no-version", r#"{ "device": {} }"#),
            ("future-version", r#"{ "schema_version": 999, "device": {} }"#),
            ("bad-device-section", r#"{ "schema_version": 1, "device": 42 }"#),
        ] {
            let path = temp_cache_file(name, content);
            assert!(AutotuneCache::load(&path).is_none(), "{name} should be rejected");
            std::fs::remove_file(&path).unwrap();
        }
        // Unreadable (missing) file: also None, no panic.
        assert!(AutotuneCache::load(Path::new("/nonexistent/tuning.json")).is_none());
    }

    /// Unknown fields and out-of-domain values warn and are skipped; the
    /// known fields of the same file still apply (forward compat).
    #[test]
    fn unknown_cache_fields_are_ignored_known_ones_apply() {
        let path = temp_cache_file(
            "unknown-fields",
            r#"{
                "schema_version": 1,
                "some_future_root_field": {},
                "device": {
                    "max_command_buffers_in_flight": 2,
                    "some_future_knob": 7,
                    "commit_every_n_dispatches": "not-a-number",
                    "moe_commit_min_routes": -1
                }
            }"#,
        );
        let cache = AutotuneCache::load(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        let t = MetalTuning::resolve(Some(&cache), None, &MetalTuningOverrides::default(), no_env);
        assert_eq!(t.max_command_buffers_in_flight, 2);
        assert_eq!(t.commit_every_n_dispatches, 0);
        assert_eq!(t.moe_commit_min_routes, 64);
    }

    /// In-flight 0 from a cache would deadlock the commit logic; it is
    /// rejected at parse time like the env var's `>= 1` filter.
    #[test]
    fn cache_in_flight_zero_is_rejected() {
        let path = temp_cache_file(
            "in-flight-zero",
            r#"{ "schema_version": 1, "device": { "max_command_buffers_in_flight": 0 } }"#,
        );
        let cache = AutotuneCache::load(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        let t = MetalTuning::resolve(Some(&cache), None, &MetalTuningOverrides::default(), no_env);
        assert_eq!(t.max_command_buffers_in_flight, 8);
    }

    #[test]
    fn sanitize_device_names() {
        assert_eq!(sanitize_device_name("Apple M4 Pro"), "apple-m4-pro");
        assert_eq!(sanitize_device_name("Apple M1 (Ultra) / 2022"), "apple-m1-ultra-2022");
        assert_eq!(sanitize_device_name("---"), "");
    }
}
