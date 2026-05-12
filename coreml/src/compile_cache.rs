//! Persistent compile cache for compiled MLPackages (`.mlmodelc` directories).
//!
//! Apple's `MLModel::compileModelAtURL` is the slow part of the load path —
//! on M1 Pro it takes 50–500 ms per subgraph for the models we've measured
//! (MobileNet, MODNet). Without a cache, every `CoremlTransform::transform`
//! call recompiles, even when re-running the same model. This module provides
//! a content-addressed disk cache of the compiled `.mlmodelc` so the second
//! and subsequent runs skip the compile step.
//!
//! ## Cache key
//!
//! `(model_bytes_sha256, weight_bytes_sha256, version_tag)` where:
//! - `model_bytes` is the prost-serialized `Model` proto (the MLPackage's
//!   `Data/com.apple.CoreML/model.mlmodel` contents)
//! - `weight_bytes` is the MILBlob v2 weight file (`weight.bin`)
//! - `version_tag` is a string baked at compile-time from `(crate_version,
//!   MIL_PROTO_VERSION_commit, OS_major)` — bumping any of these invalidates
//!   the cache without code changes
//!
//! `Manifest.json` UUIDs are NOT hashed — they're random per-write and
//! describe the package layout, not the model content.
//!
//! ## Cache location
//!
//! `$HOME/Library/Caches/tract-coreml/v1/<hash>.mlmodelc/`
//!
//! macOS' `Library/Caches` is the documented location for app caches that
//! the system may purge under disk pressure. The `v1` subdirectory holds
//! the version_tag-keyed sub-cache; bumping it (in code) invalidates without
//! deleting other versions' entries.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow};
use prost::Message;
use sha2::{Digest, Sha256};

use crate::proto::core_ml::specification as spec;
use crate::proto::core_ml::specification::mil_spec as mil;

/// Cache version tag — bump this constant when the MLPackage layout, the MIL
/// proto schema, or any other compiled-artifact-affecting input changes in a
/// way that should invalidate cached `.mlmodelc` entries crate-wide.
const CACHE_VERSION: &str = "v1";

/// Lookup key for a compiled MLPackage. Computed from the MLPackage's
/// content bytes (model.mlmodel + weight.bin); Manifest.json is excluded
/// because its UUIDs are random per-write.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheKey {
    /// 64-char hex string: `sha256(model_bytes || weight_bytes)`.
    pub digest: String,
}

impl CacheKey {
    /// Compute the cache key for the given MLPackage content bytes.
    ///
    /// **Canonical hashing.** The raw `model_bytes` from `prost::encode_to_vec`
    /// are NOT byte-stable across runs because protobuf `map<string, T>` fields
    /// (e.g. each MIL `Operation`'s `inputs` and `attributes`) serialize in
    /// `HashMap` iteration order, which is randomised per process. We work
    /// around this by re-decoding the proto and walking it in canonical
    /// (sorted-by-key) order to feed the hasher. The resulting digest is
    /// stable across runs of the same model.
    pub fn compute(model_bytes: &[u8], weight_bytes: &[u8]) -> Self {
        let mut h = Sha256::new();
        h.update(CACHE_VERSION.as_bytes());
        h.update(b"\n");
        if let Ok(model) = spec::Model::decode(model_bytes) {
            hash_model_canonical(&mut h, &model);
        } else {
            // Decode failed (very rare; would indicate a malformed proto).
            // Fall back to byte-level hash — it's at least content-sensitive
            // even if not byte-stable across runs.
            h.update(b"raw:");
            h.update(model_bytes);
        }
        h.update(b"weights:");
        h.update(weight_bytes);
        let result = h.finalize();
        let digest = result.iter().map(|b| format!("{b:02x}")).collect();
        Self { digest }
    }
}

fn hash_model_canonical(h: &mut Sha256, model: &spec::Model) {
    h.update(b"specv:");
    h.update(model.specification_version.to_le_bytes());
    if let Some(spec::model::Type::MlProgram(prog)) = &model.r#type {
        hash_program_canonical(h, prog);
    }
    if let Some(desc) = &model.description {
        hash_description_canonical(h, desc);
    }
}

fn hash_description_canonical(h: &mut Sha256, desc: &spec::ModelDescription) {
    h.update(b"desc:");
    for fd in &desc.input {
        hash_feature_description(h, b"in:", fd);
    }
    for fd in &desc.output {
        hash_feature_description(h, b"out:", fd);
    }
}

fn hash_feature_description(h: &mut Sha256, tag: &[u8], fd: &spec::FeatureDescription) {
    h.update(tag);
    h.update(fd.name.as_bytes());
    // Include the feature type (shape + dtype). Without this, two MLPackages
    // with identical MIL programs but different input shapes (e.g. MODNet
    // bound to 512×512 vs 256×256) hash to the same key and cache-collide.
    if let Some(ty) = &fd.r#type
        && let Some(spec::feature_type::Type::MultiArrayType(arr)) = &ty.r#type
    {
        h.update(b"shape:");
        for d in &arr.shape {
            h.update(d.to_le_bytes());
            h.update(b",");
        }
        h.update(b"dt:");
        h.update(arr.data_type.to_le_bytes());
    }
}

fn hash_program_canonical(h: &mut Sha256, prog: &mil::Program) {
    h.update(b"prog_v:");
    h.update(prog.version.to_le_bytes());
    let mut fn_keys: Vec<&String> = prog.functions.keys().collect();
    fn_keys.sort();
    for k in fn_keys {
        h.update(b"fn:");
        h.update(k.as_bytes());
        let f = &prog.functions[k];
        h.update(b"opset:");
        h.update(f.opset.as_bytes());
        for nvt in &f.inputs {
            h.update(b"f_in:");
            h.update(nvt.name.as_bytes());
        }
        let mut bs_keys: Vec<&String> = f.block_specializations.keys().collect();
        bs_keys.sort();
        for bk in bs_keys {
            h.update(b"bs:");
            h.update(bk.as_bytes());
            let block = &f.block_specializations[bk];
            for op in &block.operations {
                hash_op_canonical(h, op);
            }
            for o in &block.outputs {
                h.update(b"b_out:");
                h.update(o.as_bytes());
            }
        }
    }
}

fn hash_op_canonical(h: &mut Sha256, op: &mil::Operation) {
    h.update(b"op:");
    h.update(op.r#type.as_bytes());
    let mut input_keys: Vec<&String> = op.inputs.keys().collect();
    input_keys.sort();
    for k in input_keys {
        h.update(b"i:");
        h.update(k.as_bytes());
        for binding in &op.inputs[k].arguments {
            if let Some(mil::argument::binding::Binding::Name(n)) = &binding.binding {
                h.update(n.as_bytes());
                h.update(b",");
            }
        }
    }
    for o in &op.outputs {
        h.update(b"o:");
        h.update(o.name.as_bytes());
    }
    let mut attr_keys: Vec<&String> = op.attributes.keys().collect();
    attr_keys.sort();
    for k in attr_keys {
        h.update(b"a:");
        h.update(k.as_bytes());
        // Encode this single attribute's `Value` to bytes — it's a single
        // field so its serialization is stable (no sub-maps inside Value).
        let bytes = op.attributes[k].encode_to_vec();
        h.update(&bytes);
    }
}

/// Path to the per-version cache directory (e.g.
/// `~/Library/Caches/tract-coreml/v1/`). Returns `None` if the platform
/// doesn't expose a writable cache root (very rare; `$HOME` always exists
/// on macOS / iOS).
pub fn cache_dir() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    Some(PathBuf::from(home).join("Library/Caches/tract-coreml").join(CACHE_VERSION))
}

/// Path to a cached `.mlmodelc` directory for `key`. Doesn't check existence —
/// callers should follow up with [`is_cached`] or attempt to load and fall
/// back on failure.
pub fn cached_mlmodelc_path(key: &CacheKey) -> Option<PathBuf> {
    Some(cache_dir()?.join(format!("{}.mlmodelc", key.digest)))
}

/// True if a cached `.mlmodelc` exists for `key`.
pub fn is_cached(key: &CacheKey) -> bool {
    cached_mlmodelc_path(key).is_some_and(|p| p.is_dir())
}

/// Copy a freshly-compiled `.mlmodelc` directory into the cache. Returns the
/// final path inside the cache. If the cache already has an entry for `key`,
/// returns it without overwriting (assume identical content; cache-write
/// races are tolerated this way).
pub fn store(key: &CacheKey, src_mlmodelc: &Path) -> Result<PathBuf> {
    let dst =
        cached_mlmodelc_path(key).ok_or_else(|| anyhow!("compile_cache: no $HOME/Library"))?;
    if dst.exists() {
        return Ok(dst);
    }
    let cache_root = cache_dir().ok_or_else(|| anyhow!("compile_cache: no $HOME/Library"))?;
    fs::create_dir_all(&cache_root)?;

    // Atomic-ish: copy to a temp dir alongside, then rename into place.
    let tmp_dst = cache_root.join(format!(".{}.tmp", key.digest));
    if tmp_dst.exists() {
        fs::remove_dir_all(&tmp_dst)?;
    }
    copy_dir_recursive(src_mlmodelc, &tmp_dst)?;
    // rename — if another process beat us, ignore the AlreadyExists.
    if let Err(e) = fs::rename(&tmp_dst, &dst) {
        if dst.exists() {
            // Concurrent producer won the race; clean up our copy.
            let _ = fs::remove_dir_all(&tmp_dst);
        } else {
            return Err(anyhow!("compile_cache: rename {tmp_dst:?} -> {dst:?}: {e}"));
        }
    }
    Ok(dst)
}

fn copy_dir_recursive(src: &Path, dst: &Path) -> io::Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if ty.is_dir() {
            copy_dir_recursive(&from, &to)?;
        } else if ty.is_file() {
            fs::copy(&from, &to)?;
        }
        // Symlinks inside a `.mlmodelc` are unusual; skip rather than error.
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_key_is_deterministic() {
        let a = CacheKey::compute(b"hello", b"world");
        let b = CacheKey::compute(b"hello", b"world");
        assert_eq!(a, b);
        assert_eq!(a.digest.len(), 64);
    }

    #[test]
    fn cache_key_changes_with_content() {
        let a = CacheKey::compute(b"hello", b"world");
        let b = CacheKey::compute(b"hello", b"WORLD");
        assert_ne!(a, b);
        let c = CacheKey::compute(b"HELLO", b"world");
        assert_ne!(a, c);
    }
}
