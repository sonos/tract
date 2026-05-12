//! Writer for `.mlpackage` directory layout — the on-disk artifact Core ML
//! consumes via `MLModel::compileModelAtURL_error`.
//!
//! Layout:
//! ```text
//! foo.mlpackage/
//!   Manifest.json                                  (UUID-keyed item index)
//!   Data/com.apple.CoreML/
//!     model.mlmodel                                (serialized Model proto)
//!     weights/
//!       weight.bin                                 (MILBlob v2; see mil::blob)
//! ```

use std::fs;
use std::path::Path;

use serde_json::json;
use uuid::Uuid;

/// Package-relative weight file path that Core ML expects in `BlobFileValue.file_name`.
pub const WEIGHT_BLOB_PATH: &str = "@model_path/weights/weight.bin";

/// Write a complete `.mlpackage` directory to `dst`, removing any existing
/// directory at the path first.
///
/// - `model_bytes` — the prost-serialized Model proto
///   ([`crate::proto::core_ml::specification::Model`]).
/// - `weight_bytes` — the MILBlob v2 weight file (build via
///   [`crate::mil::blob::BlobBuilder`]).
pub fn write(dst: &Path, model_bytes: &[u8], weight_bytes: &[u8]) -> anyhow::Result<()> {
    if dst.exists() {
        fs::remove_dir_all(dst)?;
    }
    let data_dir = dst.join("Data/com.apple.CoreML");
    let weights_dir = data_dir.join("weights");
    fs::create_dir_all(&weights_dir)?;

    fs::write(weights_dir.join("weight.bin"), weight_bytes)?;
    fs::write(data_dir.join("model.mlmodel"), model_bytes)?;

    let manifest = build_manifest();
    fs::write(dst.join("Manifest.json"), serde_json::to_string_pretty(&manifest)?)?;

    Ok(())
}

fn build_manifest() -> serde_json::Value {
    let weights_id = Uuid::new_v4().to_string().to_uppercase();
    let model_id = Uuid::new_v4().to_string().to_uppercase();
    json!({
        "fileFormatVersion": "1.0.0",
        "itemInfoEntries": {
            weights_id: {
                "author": "com.apple.CoreML",
                "description": "CoreML Model Weights",
                "name": "weights",
                "path": "com.apple.CoreML/weights"
            },
            model_id.clone(): {
                "author": "com.apple.CoreML",
                "description": "CoreML Model Specification",
                "name": "model.mlmodel",
                "path": "com.apple.CoreML/model.mlmodel"
            }
        },
        "rootModelIdentifier": model_id,
    })
}
