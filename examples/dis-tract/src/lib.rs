//! Dis-tract: pipeline / layer-split inference for tract, aimed at running a model
//! too big for one machine.
//!
//! A worker is assigned a contiguous layer range and builds that shard itself: the
//! NNEF graph is pruned to those layers and only their weights are read
//! ([`shard_graph`]), so the full model is never materialised. It is then
//! `prepare`d for that worker's own backend (CPU / Metal / CUDA). Each shard keeps
//! its layers' KV cache resident and loops it step→step ([`llm::StageState`]); only
//! the residual activation crosses the wire, as plain host tensors over zenoh
//! ([`znet`], [`protocol`]). [`partition`] splits an in-memory model instead, for
//! the single-process reference and tests.
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
