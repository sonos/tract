//! Supertonic text to speech in the browser, on tract's CPU or WebGPU runtime, from
//! Supertone's ONNX graphs.

pub mod engine;
pub mod text;
#[cfg(target_arch = "wasm32")]
mod web;
