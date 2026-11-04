//! Bonsai Image in the browser, on tract's WebGPU runtime: three NNEF graphs carrying their
//! packed weights, read from the origin private file system one stage at a time.

mod latent_rgb;
pub mod pipeline;
#[cfg(target_arch = "wasm32")]
mod web;
