use anyhow::*;
use tract::prelude::*;

tract::impl_ndarray_interop!();

/// Flow-matching Euler scheduler for SD3.
struct FlowMatchScheduler {
    timesteps: Vec<f32>,
    sigmas: Vec<f32>,
}

impl FlowMatchScheduler {
    fn new(num_inference_steps: usize, shift: f32) -> Self {
        let n = num_inference_steps;
        // Linearly spaced sigmas from 1.0 to 0.0
        let sigmas_unshifted: Vec<f32> = (0..=n).map(|i| 1.0 - i as f32 / n as f32).collect();
        // Apply shift: sigma_shifted = shift * sigma / (1 + (shift - 1) * sigma)
        let sigmas: Vec<f32> =
            sigmas_unshifted.iter().map(|&s| shift * s / (1.0 + (shift - 1.0) * s)).collect();
        // Timesteps = sigmas[0..n] * 1000
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

const VAE_SCALING_FACTOR: f32 = 1.5305;
const VAE_SHIFT_FACTOR: f32 = 0.0609;

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

/// Stable Diffusion 3 Medium image generation with tract
#[derive(clap::Parser)]
struct Args {
    /// Text prompt
    #[arg(short, long, default_value = "a photo of a cat")]
    prompt: String,

    /// Number of images
    #[arg(short, long, default_value_t = 1)]
    num_images: usize,

    /// Denoising steps
    #[arg(short, long, default_value_t = 28)]
    steps: usize,

    /// Random seed
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Classifier-free guidance scale
    #[arg(short, long, default_value_t = 7.0)]
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
        "SD3: \"{}\", images: {num_images}, steps: {num_steps}, seed: {}, guidance: {guidance_scale}",
        args.prompt, args.seed
    );

    // --- Tokenize CLIP (shared tokenizer for CLIP-L and CLIP-G) ---
    let tokenizer = tokenizers::Tokenizer::from_file(assets.join("tokenizer/tokenizer.json"))
        .map_err(|e| anyhow!("{e}"))?;
    let encode_clip = |text: &str| -> Result<ndarray::Array2<i64>> {
        let enc = tokenizer.encode(text, true).map_err(|e| anyhow!("{e}"))?;
        let mut ids: Vec<i64> = enc.get_ids().iter().map(|&id| id as i64).collect();
        ids.resize(77, tokenizer.token_to_id("<|endoftext|>").unwrap_or(49407) as i64);
        Ok(ndarray::Array2::from_shape_vec((1, 77), ids)?)
    };
    let input_ids = encode_clip(&args.prompt)?;
    let uncond_input_ids = encode_clip("")?;

    // --- Tokenize T5 (optional, if tokenizer_3 exists) ---
    let t5_tokenizer =
        tokenizers::Tokenizer::from_file(assets.join("tokenizer_3/tokenizer.json")).ok();
    let has_t5 = t5_tokenizer.is_some() && assets.join("text_encoder_3.onnx").exists();
    let encode_t5 = |text: &str| -> Result<ndarray::Array2<i64>> {
        let tok = t5_tokenizer.as_ref().unwrap();
        let enc = tok.encode(text, true).map_err(|e| anyhow!("{e}"))?;
        let mut ids: Vec<i64> = enc.get_ids().iter().map(|&id| id as i64).collect();
        // T5 pad token is 0, max length 256
        ids.resize(256, 0);
        Ok(ndarray::Array2::from_shape_vec((1, 256), ids)?)
    };
    let t5_input_ids = if has_t5 { Some(encode_t5(&args.prompt)?) } else { None };
    let t5_uncond_ids = if has_t5 { Some(encode_t5("")?) } else { None };

    use rand::SeedableRng;
    use rand_distr::{Distribution, StandardNormal};
    let mut rng = rand::rngs::StdRng::seed_from_u64(args.seed);

    let scheduler = FlowMatchScheduler::new(num_steps, 3.0);
    let init_sigma = scheduler.init_noise_sigma();
    eprintln!("  Scheduler: {num_steps} steps, shift=3.0, init_sigma={init_sigma:.4}");

    // --- Pick runtime ---
    let gpu = ["cuda", "metal", "default"]
        .iter()
        .find_map(|rt| tract::runtime_for_name(rt).ok())
        .unwrap();
    eprintln!("Using runtime: {gpu:?}");

    // --- Load models ---
    // Text encoders run on CPU (T5-XXL is 18GB alone); transformer + VAE go on GPU.
    eprintln!("Loading models{}...", if has_t5 { " (with T5)" } else { "" });
    let onnx = tract::onnx()?;
    let cpu = tract::runtime_for_name("default")?;
    let text_encoder = cpu.prepare(onnx.load(assets.join("text_encoder.onnx"))?.into_model()?)?;
    let text_encoder_2 =
        cpu.prepare(onnx.load(assets.join("text_encoder_2.onnx"))?.into_model()?)?;
    let text_encoder_3 = if has_t5 {
        Some(cpu.prepare(onnx.load(assets.join("text_encoder_3.onnx"))?.into_model()?)?)
    } else {
        None
    };
    // --- Text encoding ---
    eprintln!("Running text encoders...");
    let cond1 = text_encoder.run([input_ids.clone().tract()?])?;
    let uncond1 = text_encoder.run([uncond_input_ids.clone().tract()?])?;
    let cond2 = text_encoder_2.run([input_ids.tract()?])?;
    let uncond2 = text_encoder_2.run([uncond_input_ids.tract()?])?;

    // CLIPTextModelWithProjection outputs:
    //   output[0] = text_embeds (pooled projected), output[1] = last_hidden_state
    let cond_h1 = cond1[1].ndarray::<f32>()?; // (1, 77, 768)
    let cond_h2 = cond2[1].ndarray::<f32>()?; // (1, 77, 1280)
    let uncond_h1 = uncond1[1].ndarray::<f32>()?;
    let uncond_h2 = uncond2[1].ndarray::<f32>()?;

    // Concatenate CLIP hidden states: (1,77,768) + (1,77,1280) → (1,77,2048), pad to 4096
    let cond_clip = ndarray::concatenate(ndarray::Axis(2), &[cond_h1.view(), cond_h2.view()])?
        .as_standard_layout()
        .into_owned();
    let uncond_clip =
        ndarray::concatenate(ndarray::Axis(2), &[uncond_h1.view(), uncond_h2.view()])?
            .as_standard_layout()
            .into_owned();

    // Pad CLIP to 4096: (1, 77, 2048) → (77 * 4096) flat
    let pad_clip = |arr: &ndarray::ArrayBase<
        ndarray::OwnedRepr<f32>,
        ndarray::Dim<ndarray::IxDynImpl>,
    >|
     -> Vec<f32> {
        let sl = arr.as_slice().unwrap();
        let mut out = Vec::with_capacity(77 * 4096);
        for token in 0..77 {
            let base = token * 2048;
            out.extend_from_slice(&sl[base..base + 2048]);
            out.extend(std::iter::repeat_n(0.0f32, 2048));
        }
        out
    };
    let cond_clip_padded = pad_clip(&cond_clip);
    let uncond_clip_padded = pad_clip(&uncond_clip);

    // Run T5 if available: (1, 256) → (1, 256, 4096)
    let (cond_t5, uncond_t5) = if let Some(te3) = &text_encoder_3 {
        eprintln!("Running T5 encoder...");
        let c = te3.run([t5_input_ids.unwrap().tract()?])?;
        let u = te3.run([t5_uncond_ids.unwrap().tract()?])?;
        let c_sl = c[0].as_slice::<f32>()?.to_vec();
        let u_sl = u[0].as_slice::<f32>()?.to_vec();
        (Some(c_sl), Some(u_sl))
    } else {
        (None, None)
    };

    // Build combined embeddings:
    //   Without T5: (seq=77, dim=4096)
    //   With T5:    (seq=77+256=333, dim=4096)
    let seq_len = if has_t5 { 77 + 256 } else { 77 };
    let b2 = 2 * num_images;
    let emb_size = seq_len * 4096;

    let build_emb = |clip_padded: &[f32], t5_data: &Option<Vec<f32>>| -> Vec<f32> {
        let mut out = Vec::with_capacity(emb_size);
        out.extend_from_slice(clip_padded);
        if let Some(t5) = t5_data {
            out.extend_from_slice(t5);
        }
        out
    };
    let cond_emb = build_emb(&cond_clip_padded, &cond_t5);
    let uncond_emb = build_emb(&uncond_clip_padded, &uncond_t5);

    let mut emb_data = Vec::with_capacity(b2 * emb_size);
    for _ in 0..num_images {
        emb_data.extend_from_slice(&uncond_emb);
    }
    for _ in 0..num_images {
        emb_data.extend_from_slice(&cond_emb);
    }
    let text_emb = ndarray::ArrayD::from_shape_vec(vec![b2, seq_len, 4096], emb_data)?;
    eprintln!("  Text embeddings: {:?}", text_emb.shape());

    // Pooled: cat CLIP-L pooled (768) + CLIP-G pooled (1280) → (1, 2048)
    let cond_p1 = cond1[0].as_slice::<f32>()?;
    let cond_p2 = cond2[0].as_slice::<f32>()?;
    let uncond_p1 = uncond1[0].as_slice::<f32>()?;
    let uncond_p2 = uncond2[0].as_slice::<f32>()?;
    let pooled_dim = cond_p1.len() + cond_p2.len(); // 768 + 1280 = 2048
    let mut pooled_data = Vec::with_capacity(b2 * pooled_dim);
    for _ in 0..num_images {
        pooled_data.extend_from_slice(uncond_p1);
        pooled_data.extend_from_slice(uncond_p2);
    }
    for _ in 0..num_images {
        pooled_data.extend_from_slice(cond_p1);
        pooled_data.extend_from_slice(cond_p2);
    }
    let pooled = ndarray::ArrayD::from_shape_vec(vec![b2, pooled_dim], pooled_data)?;

    // --- Generate latent noise (16 channels for SD3) ---
    let latent_size = 16 * 128 * 128;
    let mut latents: Vec<f32> = (0..num_images * latent_size)
        .map(|_| {
            <StandardNormal as Distribution<f32>>::sample(&StandardNormal, &mut rng) * init_sigma
        })
        .collect();

    // --- Batched denoising (load transformer, run, drop to free VRAM for VAE) ---
    eprintln!("Loading transformer...");
    let transformer = gpu.prepare(onnx.load(assets.join("transformer.onnx"))?.into_model()?)?;
    use kdam::BarExt as _;
    let mut pb = kdam::Bar::builder()
        .total(num_steps)
        .desc(format!("Denoising {num_images} image(s)"))
        .build()
        .unwrap();
    for (i, &t) in scheduler.timesteps.iter().enumerate() {
        // Build input: [uncond latents, cond latents] — no scaling needed for flow matching
        let mut sample_data = Vec::with_capacity(b2 * latent_size);
        for &x in &latents {
            sample_data.push(x);
        }
        for &x in &latents {
            sample_data.push(x);
        }
        let sample = ndarray::ArrayD::from_shape_vec(vec![b2, 16, 128, 128], sample_data)?;
        let timestep = ndarray::Array1::from_vec(vec![t; b2]).into_dyn();

        // MMDiT: 4 inputs (hidden_states, encoder_hidden_states, pooled_projections, timestep)
        let noise_pred = transformer.run(vec![
            sample.tract()?,
            text_emb.clone().tract()?,
            pooled.clone().tract()?,
            timestep.tract()?,
        ])?;
        let pred = noise_pred[0].as_slice::<f32>()?;

        // Classifier-free guidance
        let batch_latent_size = num_images * latent_size;
        let dt = scheduler.dt(i);
        for j in 0..batch_latent_size {
            let u = pred[j];
            let c = pred[batch_latent_size + j];
            let eps = u + guidance_scale * (c - u);
            latents[j] += eps * dt;
        }

        let sigma = scheduler.sigmas[i];
        pb.set_postfix(format!("t={t:.0} σ={sigma:.3}"));
        pb.update(1).ok();
    }
    eprintln!();
    drop(transformer);

    // --- VAE decode + save ---
    eprintln!("Loading VAE decoder...");
    let vae_decoder = gpu.prepare(onnx.load(assets.join("vae_decoder.onnx"))?.into_model()?)?;
    let (h, w) = (1024usize, 1024usize);
    for n in 0..num_images {
        let img_latent: Vec<f32> = latents[n * latent_size..(n + 1) * latent_size]
            .iter()
            .map(|&x| x / VAE_SCALING_FACTOR + VAE_SHIFT_FACTOR)
            .collect();
        let latent_arr = ndarray::ArrayD::from_shape_vec(vec![1, 16, 128, 128], img_latent)?;
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
