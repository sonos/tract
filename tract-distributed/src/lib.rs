//! Dis-tract: pipeline / layer-split distributed inference for tract.
//!
//! A full [`TypedModel`](tract_core::prelude::TypedModel) is split at chosen
//! activation edges into N standalone sub-models ([`partition`]). Each sub-model
//! is shipped to a worker as backend-neutral NNEF ([`codec`]), loaded, and
//! `prepare`d for that worker's own backend (CPU / Metal / CUDA). Boundary
//! activations flow stage→stage as plain host tensors over HTTP ([`protocol`]).
//!
//! This crate composes existing tract primitives; it adds no new ops.

// Force-link the backend runtimes so their `inventory` registrations are present
// in any binary depending on this lib; `runtime_for_name("metal"|"cuda")` then
// resolves. Each target links only the backend(s) valid for it.
#[cfg(any(target_os = "linux", target_os = "windows"))]
extern crate tract_cuda;
#[cfg(target_vendor = "apple")]
extern crate tract_metal;

pub mod caps;
pub mod chat;
pub mod codec;
pub mod llm;
pub mod models;
pub mod partition;
pub mod plan;
pub mod protocol;
pub mod runner;
pub mod shard_graph;
pub mod stage;
pub mod znet;
