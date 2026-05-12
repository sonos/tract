//! MIL (Model Intermediate Language) builder helpers — thin wrappers over the
//! prost-generated proto types in [`crate::proto::core_ml::specification::mil_spec`]
//! that make constructing CoreML MLPrograms in Rust ergonomic.
//!
//! Layout:
//! - [`value`] — `ValueType` and `TensorValue` constructors
//! - [`op`] — `Operation` constructors (`const` variants + arg helpers)
//! - [`blob`] — MILBlob v2 weight file writer (`BlobBuilder`)
//! - [`program`] — `Program` / `Function` / `Block` assembly helpers
//!
//! Reference for callers: see `notes/phase-1-spike.md` §"Concrete migration
//! plan" in the project root for how these compose into a CoreML transform.

pub mod blob;
pub mod op;
pub mod program;
pub mod value;
