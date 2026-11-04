//! Bonsai Image (ternary Flux2 Klein 4B) with tract.
//!
//! Prompt to PNG through the Qwen3 encoder, the ternary DiT and the Flux2 VAE exported by
//! `export.py`, each dropped before the next is loaded. `--staged` runs the load-then-free
//! stand-in instead.

use std::path::PathBuf;

use anyhow::*;
use tract::prelude::*;

tract::impl_ndarray_interop!();

mod staged;

const TEXT_LEN: usize = 256;
const TEXT_DIM: usize = 7680;
const LATENT_DIM: usize = 128;

/// FLUX.2 Klein schedule: linear sigmas under an exponential time shift, whose mu follows the
/// image token count. Klein takes the sigma itself as its timestep.
struct FlowMatchScheduler {
    sigmas: Vec<f32>,
}

impl FlowMatchScheduler {
    fn new(steps: usize, tokens: usize) -> Self {
        let tokens = tokens as f64;
        let (m200, m10) = (0.00016927 * tokens + 0.45666666, 8.73809524e-05 * tokens + 1.89833333);
        let a = (m200 - m10) / 190.0;
        let mu = (a * steps as f64 + m200 - 200.0 * a).exp();
        let mut sigmas: Vec<f32> = (0..steps)
            .map(|i| 1.0 - i as f64 * (1.0 - 1.0 / steps as f64) / (steps - 1).max(1) as f64)
            .map(|lin| (mu / (mu + (1.0 / lin - 1.0))) as f32)
            .collect();
        sigmas.push(0.0);
        FlowMatchScheduler { sigmas }
    }
}

#[derive(clap::Parser)]
struct Args {
    /// Text prompt
    #[arg(short, long, default_value = "a photo of a bonsai tree")]
    prompt: String,
    /// Denoising steps
    #[arg(short, long, default_value_t = 4)]
    steps: usize,
    /// Random seed
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Image side, as exported
    #[arg(long, default_value_t = 256)]
    size: usize,
    /// Text tokens the transformer was exported for
    #[arg(long, default_value_t = TEXT_LEN)]
    text_len: usize,
    /// Skip the text encoder and denoise from random embeddings
    #[arg(long)]
    random_text: bool,
    /// Runtime. Block-quant weights reach the GPU through wgpu only
    #[arg(long, default_value = "wgpu")]
    runtime: String,
    /// Model directory written by export.py
    #[arg(long, default_value = "assets")]
    assets: PathBuf,
    /// Decode these latent tokens (1, N, 128) from .npy instead of denoising
    #[arg(long)]
    latent: Option<PathBuf>,
    /// Output file
    #[arg(short, long, default_value = "output.png")]
    output: PathBuf,
    /// Run the staged Q4_0 / Q2_0_T / float stand-in on the wgpu runtime
    #[arg(long)]
    staged: bool,
}

/// `(1, N, C)` row-major tokens over a square grid to `(1, C, side, side)`.
fn tokens_to_latent(tokens: &Tensor) -> Result<ndarray::Array4<f32>> {
    let view = tokens.ndarray3::<f32>()?;
    let (batch, n, c) = view.dim();
    ensure!(batch == 1, "expected one image, got batch {batch}");
    let side = (n as f64).sqrt() as usize;
    ensure!(side * side == n, "{n} tokens is not a square grid");
    let channels_first = view.index_axis(ndarray::Axis(0), 0).t().as_standard_layout().into_owned();
    Ok(channels_first.into_shape_with_order((1, c, side, side))?)
}

fn save_png(image: &Tensor, path: &std::path::Path) -> Result<()> {
    let view = image.ndarray4::<f32>()?;
    let (_, channels, h, w) = view.dim();
    ensure!(channels == 3);
    let mut pixels = vec![0u8; h * w * 3];
    for y in 0..h {
        for x in 0..w {
            for ch in 0..3 {
                let val = (view[[0, ch, y, x]].clamp(-1.0, 1.0) + 1.0) / 2.0 * 255.0;
                pixels[(y * w + x) * 3 + ch] = val as u8;
            }
        }
    }
    image::save_buffer(path, &pixels, w as u32, h as u32, image::ColorType::Rgb8)?;
    Ok(())
}

/// `(input_ids, attention_mask)` for the chat-templated prompt, padded to `TEXT_LEN`.
fn tokenize(tokenizer_json: &std::path::Path, prompt: &str) -> Result<[ndarray::Array2<i64>; 2]> {
    let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_json).map_err(|e| anyhow!("{e}"))?;
    let text = format!(
        "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
    let enc = tokenizer.encode(text, false).map_err(|e| anyhow!("{e}"))?;
    let mut ids: Vec<i64> = enc.get_ids().iter().map(|&id| id as i64).collect();
    ids.truncate(TEXT_LEN);
    let mut mask = vec![1i64; ids.len()];
    let pad = tokenizer.token_to_id("<|endoftext|>").context("no pad token")? as i64;
    ids.resize(TEXT_LEN, pad);
    mask.resize(TEXT_LEN, 0);
    Ok([
        ndarray::Array2::from_shape_vec((1, TEXT_LEN), ids)?,
        ndarray::Array2::from_shape_vec((1, TEXT_LEN), mask)?,
    ])
}

fn normal(rng: &mut rand::rngs::StdRng, len: usize) -> Vec<f32> {
    use rand_distr::{Distribution, StandardNormal};
    (0..len).map(|_| <StandardNormal as Distribution<f32>>::sample(&StandardNormal, rng)).collect()
}

fn main() -> Result<()> {
    use clap::Parser as _;
    use rand::SeedableRng;
    let args = Args::parse();

    if args.staged {
        let peak = staged::run_schedule()?;
        println!("load-then-free peak resident stages: {peak}");
        return Ok(());
    }

    let gpu = tract::runtime_for_name(&args.runtime)?;
    eprintln!("runtime: {}", gpu.name()?);
    let onnx = tract::onnx()?;
    let load = |name: &str| -> Result<_> {
        let mut model = onnx.load(args.assets.join(name))?.into_model()?;
        model.transform("transformers_detect_all")?;
        gpu.prepare(model)
    };
    let mut rng = rand::rngs::StdRng::seed_from_u64(args.seed);

    let latent = if let Some(path) = &args.latent {
        tokens_to_latent(&Tensor::load_npy(path)?)?
    } else {
        let embeddings = if args.random_text {
            eprintln!("Using random text embeddings");
            let data = normal(&mut rng, args.text_len * TEXT_DIM);
            ndarray::ArrayD::from_shape_vec(vec![1, args.text_len, TEXT_DIM], data)?.tract()?
        } else {
            let [ids, mask] =
                tokenize(&args.assets.join("tokenizer/tokenizer.json"), &args.prompt)?;
            eprintln!("Running text encoder ({} tokens)...", mask.sum());
            let text_encoder = load("text_encoder.onnx")?;
            let embeddings = text_encoder.run([ids.tract()?, mask.tract()?])?[0].clone();
            let kept = embeddings.as_slice::<f32>()?[..args.text_len * TEXT_DIM].to_vec();
            ndarray::ArrayD::from_shape_vec(vec![1, args.text_len, TEXT_DIM], kept)?.tract()?
        };

        let side = args.size / 16;
        let tokens = side * side;
        let scheduler = FlowMatchScheduler::new(args.steps, tokens);
        let mut x = normal(&mut rng, tokens * LATENT_DIM);

        eprintln!("Loading transformer...");
        let transformer = load("transformer.onnx")?;
        let mut pb = kdam::tqdm!(total = args.steps, desc = "Denoising");
        for step in 0..args.steps {
            use kdam::BarExt as _;
            let (sigma, next) = (scheduler.sigmas[step], scheduler.sigmas[step + 1]);
            let hidden = ndarray::ArrayD::from_shape_vec(vec![1, tokens, LATENT_DIM], x.clone())?;
            let t = ndarray::arr1(&[sigma]);
            let out = transformer.run([hidden.tract()?, embeddings.clone(), t.tract()?])?;
            for (x, v) in x.iter_mut().zip(out[0].as_slice::<f32>()?) {
                *x += (next - sigma) * v;
            }
            pb.update(1).ok();
        }
        eprintln!();
        drop(transformer);

        let hidden = ndarray::ArrayD::from_shape_vec(vec![1, tokens, LATENT_DIM], x)?;
        tokens_to_latent(&hidden.tract()?)?
    };

    eprintln!("Loading VAE decoder...");
    let vae_decoder = load("vae_decoder.onnx")?;
    let image = vae_decoder.run([latent.tract()?])?;
    drop(vae_decoder);

    save_png(&image[0], &args.output)?;
    eprintln!("Saved {}", args.output.display());
    Ok(())
}
