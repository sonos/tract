//! Stable Diffusion 1.5 in the browser, on tract's WebGPU runtime: three NNEF graphs whose
//! weights are read from Hugging Face's own fp16 safetensors files.

pub mod pipeline;
pub mod scheduler;
pub mod tokenizer;
#[cfg(target_arch = "wasm32")]
mod web;
