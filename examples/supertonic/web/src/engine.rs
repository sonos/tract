//! Supertonic text to speech on a tract runtime: the duration predictor, the text encoder, a
//! few flow-matching steps of the vector estimator, then the vocoder. The four graphs are
//! Supertone's own ONNX files, prepared once with symbolic lengths.

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use serde::Deserialize;
use tract::prelude::*;

use crate::text;

#[derive(Deserialize)]
struct Config {
    ae: AeConfig,
    ttl: TtlConfig,
}

#[derive(Deserialize)]
struct AeConfig {
    sample_rate: usize,
    base_chunk_size: usize,
}

#[derive(Deserialize)]
struct TtlConfig {
    chunk_compress_factor: usize,
    latent_dim: usize,
}

#[derive(Deserialize)]
struct StyleJson {
    style_ttl: StyleComponent,
    style_dp: StyleComponent,
}

#[derive(Deserialize)]
struct StyleComponent {
    data: Vec<Vec<Vec<f32>>>,
    dims: Vec<usize>,
}

impl StyleComponent {
    fn tensor(&self) -> Result<Tensor> {
        let data: Vec<f32> = self.data.iter().flatten().flatten().copied().collect();
        ensure!(self.dims.len() == 3 && self.dims[0] == 1, "voice style dims {:?}", self.dims);
        Tensor::from_slice(&self.dims, &data)
    }
}

/// One voice: the style vectors the duration predictor and the text-to-latent graphs read.
pub struct Voice {
    ttl: Tensor,
    dp: Tensor,
}

impl Voice {
    pub fn from_json(json: &[u8]) -> Result<Voice> {
        let style: StyleJson = serde_json::from_slice(json).context("Parsing the voice style")?;
        Ok(Voice { ttl: style.style_ttl.tensor()?, dp: style.style_dp.tensor()? })
    }
}

/// What to say and how.
pub struct Request<'a> {
    pub text: &'a str,
    pub lang: &'a str,
    pub steps: usize,
    pub speed: f32,
    pub seed: u64,
}

/// The four graphs, prepared on one runtime, and what the text front-end needs.
pub struct Tts {
    sample_rate: usize,
    chunk_size: usize,
    latent_channels: usize,
    indexer: Vec<i64>,
    duration: Runnable,
    text_encoder: Runnable,
    vector_estimator: Runnable,
    vocoder: Runnable,
}

/// The graphs of one Supertonic release, as Supertone publishes them.
pub struct Graphs {
    pub duration_predictor: Vec<u8>,
    pub text_encoder: Vec<u8>,
    pub vector_estimator: Vec<u8>,
    pub vocoder: Vec<u8>,
    pub tts_json: Vec<u8>,
    pub unicode_indexer_json: Vec<u8>,
}

impl Tts {
    pub async fn create(runtime: &str, graphs: &Graphs) -> Result<Tts> {
        ensure!(runtime == "cpu" || runtime == "wgpu", "Unknown runtime {runtime}");
        // On wasm, looking up the wgpu runtime checks for a device, and that check cannot
        // create one. The device has to exist first.
        if runtime == "wgpu" {
            tract::wgpu::init().await?;
        }
        let config: Config =
            serde_json::from_slice(&graphs.tts_json).context("Parsing tts.json")?;
        let indexer: Vec<i64> = serde_json::from_slice(&graphs.unicode_indexer_json)
            .context("Parsing unicode_indexer.json")?;
        Ok(Tts {
            sample_rate: config.ae.sample_rate,
            chunk_size: config.ae.base_chunk_size * config.ttl.chunk_compress_factor,
            latent_channels: config.ttl.latent_dim * config.ttl.chunk_compress_factor,
            indexer,
            duration: prepare(runtime, &graphs.duration_predictor, "duration predictor")?,
            text_encoder: prepare(runtime, &graphs.text_encoder, "text encoder")?,
            vector_estimator: prepare(runtime, &graphs.vector_estimator, "vector estimator")?,
            vocoder: prepare(runtime, &graphs.vocoder, "vocoder")?,
        })
    }

    pub fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    /// Speaks one chunk of text (see [`text::chunks`]) and returns its samples.
    pub async fn speak(&self, voice: &Voice, request: &Request<'_>) -> Result<Vec<f32>> {
        ensure!(request.steps >= 1 && request.steps <= 64, "{} steps", request.steps);
        ensure!(request.speed > 0.0, "speed {}", request.speed);
        let tagged = text::prepare(request.text, request.lang);
        let ids: Vec<i64> =
            tagged.chars().map(|c| self.indexer.get(c as usize).copied().unwrap_or(-1)).collect();
        let length = ids.len();
        let ids = Tensor::from_slice(&[1, length], &ids)?;
        let text_mask = Tensor::from_slice(&[1, 1, length], &vec![1f32; length])?;

        let duration = self.duration.run([ids.clone(), voice.dp.clone(), text_mask.clone()])?;
        let seconds = host(&duration[0]).await?[0] / request.speed;
        let samples = (seconds * self.sample_rate as f32) as usize;
        let frames = samples.div_ceil(self.chunk_size).max(1);

        let text_emb = self.text_encoder.run([ids, voice.ttl.clone(), text_mask.clone()])?;
        let text_emb = text_emb.into_iter().next().context("text encoder output")?;

        let mut rng = Normal::new(request.seed);
        let noise: Vec<f32> = (0..self.latent_channels * frames).map(|_| rng.sample()).collect();
        let mut latent = Tensor::from_slice(&[1, self.latent_channels, frames], &noise)?;
        let latent_mask = Tensor::from_slice(&[1, 1, frames], &vec![1f32; frames])?;
        let total = Tensor::from_slice(&[1], &[request.steps as f32])?;
        for step in 0..request.steps {
            let current = Tensor::from_slice(&[1], &[step as f32])?;
            let out = self.vector_estimator.run([
                latent,
                text_emb.clone(),
                voice.ttl.clone(),
                latent_mask.clone(),
                text_mask.clone(),
                current,
                total.clone(),
            ])?;
            latent = out.into_iter().next().context("vector estimator output")?;
        }
        let wav = self.vocoder.run([latent])?;
        let mut wav = host(&wav[0]).await?;
        wav.truncate(samples);
        Ok(wav)
    }
}

fn prepare(runtime: &str, onnx: &[u8], what: &str) -> Result<Runnable> {
    let options = OnnxOptions::new().ignore_value_info(true).assertion("text_length>=1");
    let mut model = tract::onnx()?
        .load_buffer_with_options(onnx, options)
        .with_context(|| format!("Loading the {what}"))?;
    model.set_output_fact(0, ())?;
    let model = model.into_model().with_context(|| format!("Typing the {what}"))?;
    tract::runtime_for_name(runtime)?
        .prepare(model)
        .with_context(|| format!("Preparing the {what}"))
}

async fn host(tensor: &Tensor) -> Result<Vec<f32>> {
    let host = tensor.to_host_async().await?;
    Ok(host.as_slice::<f32>()?.to_vec())
}

/// Standard normal samples from splitmix64 and Box-Muller, so a seed gives the same voice
/// natively and in the browser.
pub struct Normal(u64);

impl Normal {
    pub fn new(seed: u64) -> Normal {
        Normal(seed)
    }

    fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn sample(&mut self) -> f32 {
        let u1 = 1.0 - self.uniform();
        let u2 = self.uniform();
        ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
    }
}
