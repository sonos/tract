use anyhow::*;
use tract::prelude::*;

tract::impl_ndarray_interop!();

/// Euler discrete scheduler — same noise schedule as SD 1.5 / SDXL.
struct EulerScheduler {
    timesteps: Vec<i64>,
    sigmas: Vec<f32>,
}

impl EulerScheduler {
    fn new(num_inference_steps: usize) -> Self {
        use ndarray::{Array1, Axis, concatenate};
        let num_train = 1000;
        let betas = Array1::linspace(0.00085f64.sqrt(), 0.012f64.sqrt(), num_train).mapv(|b| b * b);
        let alphas = betas.mapv(|b| 1.0 - b);
        let mut alphas_cumprod = alphas.clone();
        for i in 1..num_train {
            alphas_cumprod[i] *= alphas_cumprod[i - 1];
        }
        let all_sigmas = alphas_cumprod.mapv(|a| ((1.0 - a) / a).sqrt());
        let timesteps = Array1::linspace((num_train - 1) as f64, 0.0, num_inference_steps)
            .mapv(|t| t.round() as i64);
        let sigmas_at_t = timesteps.mapv(|t| {
            let t = t as f64;
            let lo = t.floor() as usize;
            let hi = (lo + 1).min(num_train - 1);
            let frac = t - lo as f64;
            (all_sigmas[lo] * (1.0 - frac) + all_sigmas[hi] * frac) as f32
        });
        let sigmas =
            concatenate(Axis(0), &[sigmas_at_t.view(), Array1::from_vec(vec![0.0f32]).view()])
                .unwrap();
        EulerScheduler { timesteps: timesteps.to_vec(), sigmas: sigmas.to_vec() }
    }

    fn init_noise_sigma(&self) -> f32 {
        self.sigmas[0]
    }

    fn scale_factor(&self, step: usize) -> f32 {
        let sigma = self.sigmas[step];
        1.0 / (sigma * sigma + 1.0).sqrt()
    }

    fn dt(&self, step: usize) -> f32 {
        self.sigmas[step + 1] - self.sigmas[step]
    }
}

const VAE_SCALING_FACTOR: f32 = 0.13025; // SDXL

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
    // Re-encode as a small PNG thumbnail to keep the escape sequence manageable.
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

/// Stable Diffusion XL 1.0 image generation with tract
#[derive(clap::Parser)]
struct Args {
    /// Text prompt
    #[arg(short, long, default_value = "a photo of a cat")]
    prompt: String,

    /// Number of images
    #[arg(short, long, default_value_t = 1)]
    num_images: usize,

    /// Denoising steps
    #[arg(short, long, default_value_t = 20)]
    steps: usize,

    /// Random seed
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Classifier-free guidance scale
    #[arg(short, long, default_value_t = 7.5)]
    guidance_scale: f32,

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
    let guidance_scale = args.guidance_scale;
    let assets = &args.assets;

    eprintln!(
        "SDXL: \"{}\", images: {num_images}, steps: {num_steps}, seed: {}, guidance: {guidance_scale}",
        args.prompt, args.seed
    );

    // --- Tokenize (same tokenizer for both text encoders) ---
    let tokenizer = tokenizers::Tokenizer::from_file(assets.join("tokenizer/tokenizer.json"))
        .map_err(|e| anyhow!("{e}"))?;
    let encode = |text: &str| -> Result<ndarray::Array2<i64>> {
        let enc = tokenizer.encode(text, true).map_err(|e| anyhow!("{e}"))?;
        let mut ids: Vec<i64> = enc.get_ids().iter().map(|&id| id as i64).collect();
        ids.resize(77, tokenizer.token_to_id("<|endoftext|>").unwrap_or(49407) as i64);
        Ok(ndarray::Array2::from_shape_vec((1, 77), ids)?)
    };
    let input_ids = encode(&args.prompt)?;
    let uncond_input_ids = encode("")?;

    use rand::SeedableRng;
    use rand_distr::{Distribution, StandardNormal};
    let mut rng = rand::rngs::StdRng::seed_from_u64(args.seed);

    let scheduler = EulerScheduler::new(num_steps);
    let init_noise_sigma = scheduler.init_noise_sigma();
    eprintln!("  Scheduler: {num_steps} steps, init_sigma={init_noise_sigma:.4}");

    // --- Pick runtime ---
    let gpu = ["cuda", "metal", "default"]
        .iter()
        .find_map(|rt| tract::runtime_for_name(rt).ok())
        .unwrap();
    eprintln!("Using runtime: {gpu:?}");

    // --- Text encoding (load each encoder, run, drop to save VRAM) ---
    eprintln!("Running text encoders...");
    let onnx = tract::onnx()?;

    let text_encoder = gpu.prepare(onnx.load(assets.join("text_encoder.onnx"))?.into_model()?)?;
    let cond1 = text_encoder.run([input_ids.clone().tract()?])?;
    let uncond1 = text_encoder.run([uncond_input_ids.clone().tract()?])?;
    drop(text_encoder);

    let text_encoder_2 =
        gpu.prepare(onnx.load(assets.join("text_encoder_2.onnx"))?.into_model()?)?;
    let cond2 = text_encoder_2.run([input_ids.tract()?])?;
    let uncond2 = text_encoder_2.run([uncond_input_ids.tract()?])?;
    drop(text_encoder_2);

    // Concatenate hidden states: (1,77,768) + (1,77,1280) → (1,77,2048)
    // TE1: output[0] = last_hidden_state (1,77,768), output[1] = pooler (1,768)
    // TE2: output[0] = text_embeds/pooled (1,1280), output[1] = last_hidden_state (1,77,1280)
    let cond_h1 = cond1[0].ndarray::<f32>()?;
    let cond_h2 = cond2[1].ndarray::<f32>()?;
    let uncond_h1 = uncond1[0].ndarray::<f32>()?;
    let uncond_h2 = uncond2[1].ndarray::<f32>()?;
    let cond_cat = ndarray::concatenate(ndarray::Axis(2), &[cond_h1.view(), cond_h2.view()])?
        .as_standard_layout()
        .into_owned();
    let uncond_cat = ndarray::concatenate(ndarray::Axis(2), &[uncond_h1.view(), uncond_h2.view()])?
        .as_standard_layout()
        .into_owned();

    // Build batched: [uncond×N, cond×N] → (2N, 77, 2048)
    let b2 = 2 * num_images;
    let emb_dim = 2048;
    let uncond_sl = uncond_cat.as_slice().unwrap();
    let cond_sl = cond_cat.as_slice().unwrap();
    let emb_size = 77 * emb_dim;
    let mut emb_data = Vec::with_capacity(b2 * emb_size);
    for _ in 0..num_images {
        emb_data.extend_from_slice(uncond_sl);
    }
    for _ in 0..num_images {
        emb_data.extend_from_slice(cond_sl);
    }
    let text_emb = ndarray::ArrayD::from_shape_vec(vec![b2, 77, emb_dim], emb_data)?;
    eprintln!("  Text embeddings: {:?}", text_emb.shape());

    // Pooled text embeddings from text_encoder_2 (output 0)
    let cond_pooled = cond2[0].as_slice::<f32>()?;
    let uncond_pooled = uncond2[0].as_slice::<f32>()?;
    let pooled_dim = cond_pooled.len();
    let mut pooled_data = Vec::with_capacity(b2 * pooled_dim);
    for _ in 0..num_images {
        pooled_data.extend_from_slice(uncond_pooled);
    }
    for _ in 0..num_images {
        pooled_data.extend_from_slice(cond_pooled);
    }
    let pooled = ndarray::ArrayD::from_shape_vec(vec![b2, pooled_dim], pooled_data)?;

    // time_ids: [original_h, original_w, crop_top, crop_left, target_h, target_w]
    let time_ids_single: Vec<f32> = vec![1024.0, 1024.0, 0.0, 0.0, 1024.0, 1024.0];
    let mut time_ids_data = Vec::with_capacity(b2 * 6);
    for _ in 0..b2 {
        time_ids_data.extend_from_slice(&time_ids_single);
    }
    let time_ids = ndarray::ArrayD::from_shape_vec(vec![b2, 6], time_ids_data)?;

    // --- Generate latent noise ---
    let latent_size = 4 * 128 * 128;
    let mut latents: Vec<f32> = (0..num_images * latent_size)
        .map(|_| {
            <StandardNormal as Distribution<f32>>::sample(&StandardNormal, &mut rng)
                * init_noise_sigma
        })
        .collect();

    // --- Batched denoising (load UNet, run, drop to free VRAM for VAE) ---
    eprintln!("Loading UNet...");
    let unet = gpu.prepare(onnx.load(assets.join("unet.onnx"))?.into_model()?)?;
    use kdam::BarExt as _;
    let mut pb = kdam::Bar::builder()
        .total(num_steps)
        .desc(format!("Denoising {num_images} image(s)"))
        .build()
        .unwrap();
    for (i, &t) in scheduler.timesteps.iter().enumerate() {
        let scale = scheduler.scale_factor(i);

        let mut sample_data = Vec::with_capacity(b2 * latent_size);
        for &x in &latents {
            sample_data.push(x * scale);
        }
        for &x in &latents {
            sample_data.push(x * scale);
        }
        let sample = ndarray::ArrayD::from_shape_vec(vec![b2, 4, 128, 128], sample_data)?;
        let timestep = ndarray::Array1::from_vec(vec![t; b2]).into_dyn();

        // SDXL UNet: 5 inputs (sample, timestep, encoder_hidden_states, time_ids, text_embeds)
        let noise_pred = unet.run(vec![
            sample.tract()?,
            timestep.tract()?,
            text_emb.clone().tract()?,
            time_ids.clone().tract()?,
            pooled.clone().tract()?,
        ])?;
        let pred = noise_pred[0].as_slice::<f32>()?;

        let batch_latent_size = num_images * latent_size;
        let dt = scheduler.dt(i);
        for j in 0..batch_latent_size {
            let u = pred[j];
            let c = pred[batch_latent_size + j];
            let eps = u + guidance_scale * (c - u);
            latents[j] += eps * dt;
        }

        let sigma = scheduler.sigmas[i];
        pb.set_postfix(format!("t={t} σ={sigma:.2}"));
        pb.update(1).ok();
    }
    eprintln!();
    drop(unet);

    // --- VAE decode + save ---
    eprintln!("Loading VAE decoder...");
    let vae_decoder = gpu.prepare(onnx.load(assets.join("vae_decoder.onnx"))?.into_model()?)?;
    let (h, w) = (1024usize, 1024usize);
    for n in 0..num_images {
        let img_latent: Vec<f32> = latents[n * latent_size..(n + 1) * latent_size]
            .iter()
            .map(|&x| x / VAE_SCALING_FACTOR)
            .collect();
        let latent_arr = ndarray::ArrayD::from_shape_vec(vec![1, 4, 128, 128], img_latent)?;
        let image_result = vae_decoder.run([latent_arr.tract()?])?;
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
