//! Text to image: the CLIP text encoder, the UNet stepped by a sampler under classifier-free
//! guidance, and the VAE decoder.

use anyhow::{Result, ensure};
use half::f16;
use tract::ReadAt;
use tract::prelude::*;

use crate::scheduler::{Noise, Sampler, Scheduler};
use crate::tokenizer::{CONTEXT, Tokenizer};

const VAE_SCALING_FACTOR: f32 = 0.18215;
pub const SIZE: usize = 512;
const LATENT: usize = 4 * 64 * 64;

/// One of the three graphs, on the WebGPU runtime: an NNEF tar of the graph, its weights read
/// from the safetensors file `weights` reads, each when the runtime first needs it.
pub fn load(graph_tar: &[u8], weights: impl ReadAt + 'static) -> Result<Runnable> {
    let mut model = tract::nnef()?.load_buffer(graph_tar)?;
    model.transform("transformers_detect_all")?;
    model.link_safetensors(weights)?;
    tract::runtime_for_name("wgpu")?.prepare(model)
}

/// The outputs of `model` on `inputs`, read back from the GPU.
async fn run(model: &Runnable, inputs: Vec<Tensor>) -> Result<Vec<Tensor>> {
    let mut outputs = vec![];
    for output in model.run(inputs)? {
        outputs.push(output.to_host_async().await?);
    }
    Ok(outputs)
}

fn floats(tensor: &Tensor) -> Result<Vec<f32>> {
    Ok(tensor.convert_to(DatumType::F32)?.as_slice::<f32>()?.to_vec())
}

pub enum Progress {
    Encoding,
    Step(usize, usize),
    Decoding,
}

pub struct Pipeline {
    pub tokenizer: Tokenizer,
    pub text_encoder: Runnable,
    pub unet: Runnable,
    pub vae: Runnable,
}

pub struct Request<'a> {
    pub prompt: &'a str,
    /// What to steer away from: the unconditioned half of classifier-free guidance.
    pub negative: &'a str,
    pub steps: usize,
    pub seed: u64,
    pub guidance: f32,
    pub sampler: Sampler,
}

impl Pipeline {
    /// A 512x512 RGBA image for `request`.
    pub async fn generate(
        &self,
        request: &Request<'_>,
        progress: &mut dyn FnMut(Progress),
    ) -> Result<Vec<u8>> {
        ensure!(request.steps >= 2, "at least two steps");
        progress(Progress::Encoding);
        let mut ids = self.tokenizer.encode(request.negative)?;
        ids.extend(self.tokenizer.encode(request.prompt)?);
        let ids = Tensor::from_slice(&[2, CONTEXT], &ids)?;
        let embeddings = run(&self.text_encoder, vec![ids]).await?.remove(0);
        let embeddings = embeddings.convert_to(DatumType::F16)?;

        let mut scheduler = Scheduler::new(request.sampler, request.steps);
        let sigma0 = scheduler.init_noise_sigma();
        let mut latents: Vec<f32> =
            Noise::new(request.seed).samples(LATENT).into_iter().map(|x| x * sigma0).collect();
        for (step, t) in scheduler.timesteps.clone().into_iter().enumerate() {
            progress(Progress::Step(step, request.steps));
            let scale = scheduler.input_scale(step);
            let sample: Vec<f16> =
                latents.iter().chain(latents.iter()).map(|x| f16::from_f32(x * scale)).collect();
            let inputs = vec![
                Tensor::from_slice(&[2, 4, 64, 64], &sample)?,
                Tensor::from_slice(&[2], &[t, t])?,
                embeddings.clone(),
            ];
            let noise = floats(&run(&self.unet, inputs).await?.remove(0))?;
            let (uncond, cond) = noise.split_at(LATENT);
            let eps: Vec<f32> =
                uncond.iter().zip(cond).map(|(u, c)| u + request.guidance * (c - u)).collect();
            scheduler.step(step, &mut latents, &eps);
        }

        progress(Progress::Decoding);
        let latent: Vec<f16> =
            latents.iter().map(|x| f16::from_f32(x / VAE_SCALING_FACTOR)).collect();
        let latent = Tensor::from_slice(&[1, 4, 64, 64], &latent)?;
        let image = floats(&run(&self.vae, vec![latent]).await?.remove(0))?;
        let plane = SIZE * SIZE;
        let mut rgba = vec![255u8; plane * 4];
        for p in 0..plane {
            for c in 0..3 {
                rgba[p * 4 + c] =
                    ((image[c * plane + p].clamp(-1.0, 1.0) + 1.0) * 127.5).round() as u8;
            }
        }
        Ok(rgba)
    }
}
