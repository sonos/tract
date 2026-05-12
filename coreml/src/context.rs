//! Wrapper around `objc2-core-ml`'s `MLModel`.
//!
//! Owns the loaded `MLModel` and the URL of its compiled `.mlmodelc` artifact.
//! Compilation happens at construction (via `MLModel::compileModelAtURL_error`).

use std::path::Path;
use std::sync::Mutex;

use anyhow::{Result, anyhow};
use objc2::AllocAnyThread;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_core_ml::{
    MLComputeUnits, MLDictionaryFeatureProvider, MLFeatureProvider, MLModel, MLModelConfiguration,
};
use objc2_foundation::{NSDictionary, NSError, NSString, NSURL};

/// Wraps a loaded `MLModel`.
///
/// `MLModel` is documented as thread-safe for prediction but not for
/// configuration changes (apple-coreml-developer-docs.md §8). We hold it inside
/// a `Mutex` to make the threading semantics unambiguous: predictions serialize
/// through the lock, and any future configuration changes are also safe. Cost
/// is negligible because Core ML internally serializes ANE dispatch anyway.
///
/// # Safety: `Send` + `Sync`
///
/// `Retained<MLModel>` is `!Send + !Sync` in `objc2-core-ml` 0.3 because Obj-C
/// objects are conservative-by-default about cross-thread access. Apple
/// documents `MLModel` as thread-safe for prediction, and our `Mutex` covers
/// the rest. The `unsafe impl`s below are justified by that combination.
pub struct CoremlContext {
    model: Mutex<Retained<MLModel>>,
}

unsafe impl Send for CoremlContext {}
unsafe impl Sync for CoremlContext {}

impl std::fmt::Debug for CoremlContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CoremlContext").finish_non_exhaustive()
    }
}

impl CoremlContext {
    /// Compile a `.mlpackage` (Core ML's runtime API; no Xcode tools required)
    /// and load the resulting `.mlmodelc` for inference with the given compute
    /// units.
    ///
    /// Compilation result is NOT cached by this entry point — see
    /// [`Self::load_mlpackage_cached`] for the cache-aware variant.
    pub fn load_mlpackage(path: &Path, compute_units: MLComputeUnits) -> Result<Self> {
        let compiled_url = compile_mlpackage(path)?;
        Self::load_compiled(&compiled_url, compute_units, path)
    }

    /// Load a pre-compiled `.mlmodelc` directory directly (skip the
    /// `compileModelAtURL` step). Used by the persistent compile cache to
    /// load cached entries.
    pub fn load_mlmodelc(path: &Path, compute_units: MLComputeUnits) -> Result<Self> {
        let path_str = path.to_string_lossy();
        let path_ns = NSString::from_str(&path_str);
        let url = NSURL::fileURLWithPath(&path_ns);
        Self::load_compiled(&url, compute_units, path)
    }

    /// Compile + cache flow:
    /// 1. Hash the MLPackage's content (model.mlmodel + weight.bin) to form a
    ///    cache key.
    /// 2. If `~/Library/Caches/tract-coreml/v1/<key>.mlmodelc/` exists, load
    ///    it directly (skip the `compileModelAtURL` step).
    /// 3. Otherwise: write the .mlpackage to `pkg_path`, compile it, copy the
    ///    resulting `.mlmodelc` into the cache, and load it.
    ///
    /// `pkg_path` is the path the .mlpackage will be (or already is) written
    /// to on a cache miss. On a hit, the .mlpackage doesn't need to be
    /// written at all.
    pub fn load_mlpackage_cached(
        pkg_path: &Path,
        model_bytes: &[u8],
        weight_bytes: &[u8],
        compute_units: MLComputeUnits,
        write_pkg: impl FnOnce(&Path) -> Result<()>,
    ) -> Result<(Self, CacheStatus)> {
        let key = crate::compile_cache::CacheKey::compute(model_bytes, weight_bytes);
        if let Some(cached) = crate::compile_cache::cached_mlmodelc_path(&key)
            && cached.is_dir()
        {
            // Cache hit. Skip writing the .mlpackage entirely.
            let ctx = Self::load_mlmodelc(&cached, compute_units)?;
            return Ok((ctx, CacheStatus::Hit));
        }
        // Cache miss. Write the .mlpackage, compile, copy compiled output
        // into the cache, then load.
        write_pkg(pkg_path)?;
        let compiled_url = compile_mlpackage(pkg_path)?;

        // The compiled URL is owned by macOS' temp dir — copy it into our
        // cache before it gets GC'd. Convert NSURL → PathBuf via the path()
        // accessor.
        let compiled_path = nsurl_to_pathbuf(&compiled_url)
            .ok_or_else(|| anyhow!("compiled URL has no file system path"))?;
        let cached = crate::compile_cache::store(&key, &compiled_path)?;

        let ctx = Self::load_mlmodelc(&cached, compute_units)?;
        Ok((ctx, CacheStatus::Miss))
    }

    fn load_compiled(
        compiled_url: &NSURL,
        compute_units: MLComputeUnits,
        diag_path: &Path,
    ) -> Result<Self> {
        let config = unsafe { MLModelConfiguration::new() };
        unsafe { config.setComputeUnits(compute_units) };
        let model = unsafe {
            MLModel::modelWithContentsOfURL_configuration_error(compiled_url, &config)
                .map_err(|e| anyhow!("MLModel load({diag_path:?}) failed: {}", ns_error_msg(&e)))?
        };
        Ok(Self { model: Mutex::new(model) })
    }

    /// Run a prediction with a pre-built input feature provider. Caller is
    /// responsible for constructing the `MLDictionaryFeatureProvider` (typically
    /// via [`crate::tensor`] helpers).
    pub fn predict(
        &self,
        input: &MLDictionaryFeatureProvider,
    ) -> Result<Retained<ProtocolObject<dyn MLFeatureProvider>>> {
        let provider_proto: &ProtocolObject<dyn MLFeatureProvider> =
            ProtocolObject::from_ref(input);

        let guard = self.model.lock().expect("CoremlContext mutex poisoned");
        let out = unsafe {
            guard
                .predictionFromFeatures_error(provider_proto)
                .map_err(|e| anyhow!("prediction failed: {}", ns_error_msg(&e)))?
        };
        Ok(out)
    }

    /// Build an `MLDictionaryFeatureProvider` from a slice of `(name, AnyObject)`
    /// pairs (typically `(name, MLFeatureValue)`). Convenience for callers that
    /// already have feature values constructed.
    pub fn make_input_provider(
        named_features: &[(&str, &objc2::runtime::AnyObject)],
    ) -> Result<Retained<MLDictionaryFeatureProvider>> {
        let key_strs: Vec<Retained<NSString>> =
            named_features.iter().map(|(n, _)| NSString::from_str(n)).collect();
        let key_refs: Vec<&NSString> = key_strs.iter().map(|s| &**s).collect();
        let val_refs: Vec<&objc2::runtime::AnyObject> =
            named_features.iter().map(|(_, v)| *v).collect();

        let dict: Retained<NSDictionary<NSString, objc2::runtime::AnyObject>> =
            NSDictionary::from_slices(&key_refs, &val_refs);
        let alloc = MLDictionaryFeatureProvider::alloc();
        let provider = unsafe {
            MLDictionaryFeatureProvider::initWithDictionary_error(alloc, &dict).map_err(|e| {
                anyhow!("MLDictionaryFeatureProvider init failed: {}", ns_error_msg(&e))
            })?
        };
        Ok(provider)
    }
}

fn ns_error_msg(e: &NSError) -> String {
    e.localizedDescription().to_string()
}

/// Whether [`CoremlContext::load_mlpackage_cached`] hit or missed the cache.
/// Useful for diagnostics + tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheStatus {
    Hit,
    Miss,
}

/// Run `MLModel::compileModelAtURL` on a `.mlpackage` directory. Returns the
/// NSURL of the compiled `.mlmodelc` (typically in macOS' per-process temp
/// dir; caller is responsible for caching if persistence is needed).
fn compile_mlpackage(path: &Path) -> Result<Retained<NSURL>> {
    let path_str = path.to_string_lossy();
    let path_ns = NSString::from_str(&path_str);
    let pkg_url = NSURL::fileURLWithPath(&path_ns);
    #[allow(deprecated)]
    let compiled_url = unsafe {
        MLModel::compileModelAtURL_error(&pkg_url)
            .map_err(|e| anyhow!("compileModelAtURL({path:?}) failed: {}", ns_error_msg(&e)))?
    };
    Ok(compiled_url)
}

fn nsurl_to_pathbuf(url: &NSURL) -> Option<std::path::PathBuf> {
    let path_ns = url.path()?;
    Some(std::path::PathBuf::from(path_ns.to_string()))
}
