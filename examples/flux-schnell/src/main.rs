use anyhow::*;
use tract::prelude::*;

/// Flow-matching Euler scheduler for Flux.
struct FlowMatchScheduler {
    timesteps: Vec<f32>,
    sigmas: Vec<f32>,
}

impl FlowMatchScheduler {
    fn new(num_inference_steps: usize, shift: f32) -> Self {
        let n = num_inference_steps;
        let sigmas_unshifted: Vec<f32> = (0..=n).map(|i| 1.0 - i as f32 / n as f32).collect();
        let sigmas: Vec<f32> =
            sigmas_unshifted.iter().map(|&s| shift * s / (1.0 + (shift - 1.0) * s)).collect();
        let timesteps: Vec<f32> = sigmas[..n].iter().map(|&s| s * 1000.0).collect();
        FlowMatchScheduler { timesteps, sigmas }
    }

    fn init_noise_sigma(&self) -> f32 {
        self.sigmas[0]
    }

    fn dt(&self, step: usize) -> f32 {
        self.sigmas[step + 1] - self.sigmas[step]
    }
}

const VAE_SCALING_FACTOR: f32 = 0.3611;
const VAE_SHIFT_FACTOR: f32 = 0.1159;

fn base64_encode(data: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity((data.len() + 2) / 3 * 4);
    for chunk in data.chunks(3) {
        let b = match chunk.len() {
            3 => [chunk[0], chunk[1], chunk[2]],
            2 => [chunk[0], chunk[1], 0],
            _ => [chunk[0], 0, 0],
        };
        let n = (b[0] as u32) << 16 | (b[1] as u32) << 8 | b[2] as u32;
        out.push(CHARS[(n >> 18 & 63) as usize] as char);
        out.push(CHARS[(n >> 12 & 63) as usize] as char);
        out.push(if chunk.len() > 1 { CHARS[(n >> 6 & 63) as usize] as char } else { '=' });
        out.push(if chunk.len() > 2 { CHARS[(n & 63) as usize] as char } else { '=' });
    }
    out
}

fn display_inline(path: &std::path::Path) {
    use std::io::Write as _;
    let img = match image::open(path) {
        std::result::Result::Ok(img) => img,
        _ => return,
    };
    let thumb = img.thumbnail(256, 256);
    let mut png_data = Vec::new();
    let encoder = image::codecs::png::PngEncoder::new(std::io::Cursor::new(&mut png_data));
    if thumb.write_with_encoder(encoder).is_err() {
        return;
    }
    let b64 = base64_encode(&png_data);
    let in_tmux = std::env::var("TMUX").is_ok();
    let osc = if in_tmux { "\x1bPtmux;\x1b\x1b]" } else { "\x1b]" };
    let st = if in_tmux { "\x07\x1b\\" } else { "\x07" };
    let _ = write!(
        std::io::stderr(),
        "{osc}1337;File=inline=1;width=20;preserveAspectRatio=1:{b64}{st}\n"
    );
}

/// FLUX.1-schnell image generation with tract
#[derive(clap::Parser)]
struct Args {
    /// Text prompt
    #[arg(short, long, default_value = "a photo of a cat")]
    prompt: String,

    /// Number of images
    #[arg(short, long, default_value_t = 1)]
    num_images: usize,

    /// Denoising steps (Schnell works well with 4)
    #[arg(short, long, default_value_t = 4)]
    steps: usize,

    /// Random seed
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Output filename
    #[arg(short, long, default_value = "output.png")]
    output: String,

    /// Model assets directory
    #[arg(long, default_value = "assets")]
    assets: std::path::PathBuf,
}

fn main() -> Result<()> {
    use clap::Parser as _;
    let args = Args::parse();
    let num_images = args.num_images;
    let num_steps = args.steps;
    let assets = &args.assets;

    eprintln!(
        "Flux-Schnell: \"{}\", images: {num_images}, steps: {num_steps}, seed: {}",
        args.prompt, args.seed
    );

    // --- Tokenize CLIP ---
    let clip_tokenizer = tokenizers::Tokenizer::from_file(assets.join("tokenizer/tokenizer.json"))
        .map_err(|e| anyhow!("{e}"))?;
    let encode_clip = |text: &str| -> Result<ndarray::Array2<i64>> {
        let enc = clip_tokenizer.encode(text, true).map_err(|e| anyhow!("{e}"))?;
        let mut ids: Vec<i64> = enc.get_ids().iter().map(|&id| id as i64).collect();
        ids.resize(77, clip_tokenizer.token_to_id("<|endoftext|>").unwrap_or(49407) as i64);
        Ok(ndarray::Array2::from_shape_vec((1, 77), ids)?)
    };
    let clip_ids = encode_clip(&args.prompt)?;

    // --- Tokenize T5 ---
    let t5_tokenizer = tokenizers::Tokenizer::from_file(assets.join("tokenizer_2/tokenizer.json"))
        .map_err(|e| anyhow!("{e}"))?;
    let encode_t5 = |text: &str| -> Result<ndarray::Array2<i64>> {
        let enc = t5_tokenizer.encode(text, true).map_err(|e| anyhow!("{e}"))?;
        let mut ids: Vec<i64> = enc.get_ids().iter().map(|&id| id as i64).collect();
        ids.resize(256, 0); // T5 pad token = 0
        Ok(ndarray::Array2::from_shape_vec((1, 256), ids)?)
    };
    let t5_ids = encode_t5(&args.prompt)?;

    use rand::SeedableRng;
    use rand_distr::{Distribution, StandardNormal};
    let mut rng = rand::rngs::StdRng::seed_from_u64(args.seed);

    let scheduler = FlowMatchScheduler::new(num_steps, 1.0);
    let init_sigma = scheduler.init_noise_sigma();
    eprintln!("  Scheduler: {num_steps} steps, shift=1.0, init_sigma={init_sigma:.4}");

    // --- Pick runtime ---
    let gpu = ["cuda", "metal", "default"]
        .iter()
        .find_map(|rt| tract::runtime_for_name(rt).ok())
        .unwrap();
    eprintln!("Using runtime: {gpu:?}");

    // --- Load models ---
    // Text encoders on CPU (T5-XXL is 18GB fp32), transformer + VAE on GPU.
    eprintln!("Loading models...");
    let onnx = tract::onnx()?;
    let cpu = tract::runtime_for_name("default")?;
    let clip_encoder = cpu.prepare(onnx.load(assets.join("text_encoder.onnx"))?.into_model()?)?;
    let t5_encoder = cpu.prepare(onnx.load(assets.join("text_encoder_2.onnx"))?.into_model()?)?;
    let transformer = gpu.prepare(onnx.load(assets.join("transformer.onnx"))?.into_model()?)?;
    let vae_decoder = gpu.prepare(onnx.load(assets.join("vae_decoder.onnx"))?.into_model()?)?;

    // --- Text encoding ---
    eprintln!("Running text encoders...");
    let clip_out = clip_encoder.run([clip_ids])?;
    // CLIP: output[0] = text_embeds (pooled), output[1] = last_hidden_state
    let pooled = clip_out[0].view::<f32>()?;
    let pooled_sl = pooled.as_slice().unwrap();

    eprintln!("Running T5 encoder...");
    let t5_out = t5_encoder.run([t5_ids])?;
    let t5_embeds = t5_out[0].view::<f32>()?;
    let t5_sl = t5_embeds.as_slice().unwrap();

    // No CFG for Schnell — single batch per image
    let pooled_dim = pooled_sl.len(); // 768

    // --- Generate latent noise (16 channels) ---
    let latent_size = 16 * 128 * 128;
    let mut latents: Vec<f32> = (0..num_images * latent_size)
        .map(|_| {
            <StandardNormal as Distribution<f32>>::sample(&StandardNormal, &mut rng) * init_sigma
        })
        .collect();

    // --- Denoising (no CFG, one image at a time) ---
    use kdam::BarExt as _;
    let mut pb = kdam::Bar::builder()
        .total(num_steps * num_images)
        .desc(format!("Denoising {num_images} image(s)"))
        .build()
        .unwrap();

    for n in 0..num_images {
        let offset = n * latent_size;
        // Per-image T5 and pooled embeddings
        let t5_arr = tract_ndarray::ArrayD::from_shape_vec(vec![1, 256, 4096], t5_sl.to_vec())?;
        let pooled_arr =
            tract_ndarray::ArrayD::from_shape_vec(vec![1, pooled_dim], pooled_sl.to_vec())?;

        for (i, &t) in scheduler.timesteps.iter().enumerate() {
            let sample = tract_ndarray::ArrayD::from_shape_vec(
                vec![1, 16, 128, 128],
                latents[offset..offset + latent_size].to_vec(),
            )?;
            let timestep = tract_ndarray::Array1::from_vec(vec![t]).into_dyn();

            let pred = transformer.run(vec![
                tensor(sample)?,
                tensor(t5_arr.clone())?,
                tensor(pooled_arr.clone())?,
                tensor(timestep)?,
            ])?;
            let pred_sl = pred[0].as_slice::<f32>()?;

            let dt = scheduler.dt(i);
            for j in 0..latent_size {
                latents[offset + j] += pred_sl[j] * dt;
            }

            let sigma = scheduler.sigmas[i];
            pb.set_postfix(format!("img={n} t={t:.0} σ={sigma:.3}"));
            pb.update(1).ok();
        }
    }
    eprintln!();

    // --- VAE decode + save ---
    let (h, w) = (1024usize, 1024usize);
    for n in 0..num_images {
        let offset = n * latent_size;
        let img_latent: Vec<f32> = latents[offset..offset + latent_size]
            .iter()
            .map(|&x| x / VAE_SCALING_FACTOR + VAE_SHIFT_FACTOR)
            .collect();
        let latent_arr = tract_ndarray::ArrayD::from_shape_vec(vec![1, 16, 128, 128], img_latent)?;
        let image_result = vae_decoder.run([latent_arr])?;
        let image_data = image_result[0].as_slice::<f32>()?;

        let mut pixels = vec![0u8; h * w * 3];
        for y in 0..h {
            for x in 0..w {
                for ch in 0..3 {
                    let val =
                        (image_data[ch * h * w + y * w + x].clamp(-1.0, 1.0) + 1.0) / 2.0 * 255.0;
                    pixels[(y * w + x) * 3 + ch] = val as u8;
                }
            }
        }
        let path = if num_images == 1 {
            std::path::PathBuf::from(&args.output)
        } else {
            let stem = std::path::Path::new(&args.output)
                .file_stem()
                .unwrap_or_default()
                .to_str()
                .unwrap_or("output");
            let ext = std::path::Path::new(&args.output)
                .extension()
                .unwrap_or_default()
                .to_str()
                .unwrap_or("png");
            std::path::PathBuf::from(format!("{stem}_{n}.{ext}"))
        };
        image::save_buffer(&path, &pixels, w as u32, h as u32, image::ColorType::Rgb8)?;
        eprintln!("Saved {}", path.display());
        display_inline(&path);
    }

    Ok(())
}
