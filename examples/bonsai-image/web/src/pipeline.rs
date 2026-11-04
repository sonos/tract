//! Text to image: Qwen3, the ternary DiT stepped by flow matching, and the Flux2 VAE.

use std::io::Read;

use anyhow::{Context, Result, ensure};
use tract::prelude::*;

use crate::latent_rgb::LATENT_RGB;

/// Text tokens the encoder and the transformer take: enough for a prompt.
pub const TEXT_LEN: usize = 64;
const TEXT_DIM: usize = 7680;
const LATENT_DIM: usize = 128;

/// One of the three graphs on the WebGPU runtime, from a stream of its NNEF tar.
pub fn load(tar: &mut dyn Read) -> Result<Runnable> {
    let mut nnef = tract::nnef()?;
    nnef.enable_tract_transformers()?;
    let mut model = nnef.load_read(tar)?;
    model.transform("transformers_detect_all")?;
    tract::runtime_for_name("wgpu")?.prepare(model)
}

/// The first output of `model` on `inputs`, read back from the GPU.
async fn run(model: &Runnable, inputs: Vec<Tensor>) -> Result<Tensor> {
    let output = model.run(inputs)?.into_iter().next().context("no output")?;
    output.to_host_async().await
}

/// FLUX.2 Klein schedule: linear sigmas under an exponential time shift, whose mu follows the
/// image token count. Klein takes the sigma itself as its timestep.
fn sigmas(steps: usize, tokens: usize) -> Vec<f32> {
    let tokens = tokens as f64;
    let (m200, m10) = (0.00016927 * tokens + 0.45666666, 8.73809524e-05 * tokens + 1.89833333);
    let a = (m200 - m10) / 190.0;
    let mu = (a * steps as f64 + m200 - 200.0 * a).exp();
    let mut sigmas: Vec<f32> = (0..steps)
        .map(|i| 1.0 - i as f64 * (1.0 - 1.0 / steps as f64) / (steps - 1).max(1) as f64)
        .map(|lin| (mu / (mu + (1.0 / lin - 1.0))) as f32)
        .collect();
    sigmas.push(0.0);
    sigmas
}

/// Standard normal samples from a seed: SplitMix64 through Box-Muller.
struct Noise(u64);

impl Noise {
    fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn samples(&mut self, n: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(n + 1);
        while out.len() < n {
            let (u1, u2) = (self.uniform(), self.uniform());
            let r = (-2.0 * u1.ln()).sqrt();
            let a = std::f64::consts::TAU * u2;
            out.push((r * a.cos()) as f32);
            out.push((r * a.sin()) as f32);
        }
        out.truncate(n);
        out
    }
}

/// `input_ids` and `attention_mask` for the chat-templated prompt, padded to `TEXT_LEN`.
pub fn tokenize(tokenizer: &tokenizers::Tokenizer, prompt: &str) -> Result<[Tensor; 2]> {
    let text = format!(
        "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
    let enc = tokenizer.encode(text, false).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut ids: Vec<i64> = enc.get_ids().iter().map(|&id| id as i64).collect();
    ids.truncate(TEXT_LEN);
    let mut mask = vec![1i64; ids.len()];
    let pad = tokenizer.token_to_id("<|endoftext|>").context("no pad token")? as i64;
    ids.resize(TEXT_LEN, pad);
    mask.resize(TEXT_LEN, 0);
    Ok([Tensor::from_slice(&[1, TEXT_LEN], &ids)?, Tensor::from_slice(&[1, TEXT_LEN], &mask)?])
}

pub enum Progress {
    Loading(&'static str),
    Encoding,
    Step(usize),
    /// What step `0` makes of the image: RGBA, one pixel per token.
    Preview(usize, Vec<u8>),
    Decoding,
}

/// The image `x0` holds in latent tokens, one RGBA pixel per token.
fn preview(x0: &[f32]) -> Vec<u8> {
    let mut rgba = Vec::with_capacity(x0.len() / LATENT_DIM * 4);
    for token in x0.chunks_exact(LATENT_DIM) {
        for row in &LATENT_RGB {
            let mix = token.iter().zip(row).map(|(t, w)| t * w).sum::<f32>() + row[LATENT_DIM];
            rgba.push(((mix.clamp(-1.0, 1.0) + 1.0) * 127.5) as u8);
        }
        rgba.push(255);
    }
    rgba
}

/// `(1, N, C)` row-major tokens over a `grid_h` by `grid_w` grid to `(1, C, grid_h, grid_w)`.
fn tokens_to_latent(tokens: &[f32], grid_h: usize, grid_w: usize) -> Result<Tensor> {
    let count = grid_h * grid_w;
    let mut latent = vec![0f32; tokens.len()];
    for (n, token) in tokens.chunks_exact(LATENT_DIM).enumerate() {
        for (c, v) in token.iter().enumerate() {
            latent[c * count + n] = *v;
        }
    }
    Tensor::from_slice(&[1, LATENT_DIM, grid_h, grid_w], &latent)
}

/// A `width`x`height` RGBA image for `prompt`, both multiples of 16. `stage` hands out each model
/// by name.
#[allow(clippy::too_many_arguments)]
pub async fn generate(
    tokenizer: &tokenizers::Tokenizer,
    prompt: &str,
    steps: usize,
    seed: u64,
    width: usize,
    height: usize,
    stage: &mut dyn FnMut(&'static str) -> Result<Runnable>,
    progress: &mut dyn FnMut(Progress),
) -> Result<Vec<u8>> {
    ensure!(steps >= 1, "at least one step");
    let [ids, mask] = tokenize(tokenizer, prompt)?;

    progress(Progress::Loading("text encoder"));
    let text_encoder = stage("text_encoder")?;
    progress(Progress::Encoding);
    let embeddings = run(&text_encoder, vec![ids, mask]).await?;
    ensure!(embeddings.shape()? == [1, TEXT_LEN, TEXT_DIM], "unexpected embeddings shape");

    progress(Progress::Loading("transformer"));
    let transformer = stage("transformer")?;
    let (grid_h, grid_w) = (height / 16, width / 16);
    let tokens = grid_h * grid_w;
    let sigmas = sigmas(steps, tokens);
    let mut x = Noise(seed).samples(tokens * LATENT_DIM);
    for step in 0..steps {
        progress(Progress::Step(step));
        let (sigma, next) = (sigmas[step], sigmas[step + 1]);
        let inputs = vec![
            Tensor::from_slice(&[1, tokens, LATENT_DIM], &x)?,
            embeddings.clone(),
            Tensor::from_slice(&[1], &[sigma])?,
        ];
        let velocity = run(&transformer, inputs).await?;
        let velocity = velocity.as_slice::<f32>()?;
        let x0: Vec<f32> = x.iter().zip(velocity).map(|(x, v)| x - sigma * v).collect();
        progress(Progress::Preview(step, preview(&x0)));
        for (x, v) in x.iter_mut().zip(velocity) {
            *x += (next - sigma) * v;
        }
    }

    progress(Progress::Loading("image decoder"));
    let vae_decoder = stage("vae_decoder")?;
    progress(Progress::Decoding);
    let image = run(&vae_decoder, vec![tokens_to_latent(&x, grid_h, grid_w)?]).await?;

    let image = image.as_slice::<f32>()?;
    let plane = width * height;
    let mut rgba = vec![255u8; plane * 4];
    for p in 0..plane {
        for c in 0..3 {
            rgba[p * 4 + c] = ((image[c * plane + p].clamp(-1.0, 1.0) + 1.0) * 127.5).round() as u8;
        }
    }
    Ok(rgba)
}
