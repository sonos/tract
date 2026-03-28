use anyhow::*;
use tract::prelude::*;

/// SD 1.5 Euler discrete scheduler — compute sigmas from scratch.
struct EulerScheduler {
    timesteps: Vec<i64>,
    sigmas: Vec<f32>,
}

impl EulerScheduler {
    fn new(num_inference_steps: usize) -> Self {
        use tract_ndarray::{Array1, Axis, concatenate};

        // SD 1.5 config: scaled_linear beta schedule, 1000 training steps
        let num_train = 1000;
        let betas = Array1::linspace(0.00085f64.sqrt(), 0.012f64.sqrt(), num_train).mapv(|b| b * b);
        let alphas = betas.mapv(|b| 1.0 - b);

        // cumulative product -> sigmas
        let mut alphas_cumprod = alphas.clone();
        for i in 1..num_train {
            alphas_cumprod[i] *= alphas_cumprod[i - 1];
        }
        let all_sigmas = alphas_cumprod.mapv(|a| ((1.0 - a) / a).sqrt());

        // Timesteps: linspace num_train-1 -> 0
        let timesteps = Array1::linspace((num_train - 1) as f64, 0.0, num_inference_steps)
            .mapv(|t| t.round() as i64);

        // Interpolate sigmas at timestep positions, append 0
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

/// SD 1.5 VAE scaling factor
const VAE_SCALING_FACTOR: f32 = 0.18215;

fn main() -> Result<()> {
    let assets = std::path::Path::new("assets");
    let num_images: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(1);

    // Tokenize prompt
    let prompt = "a photo of a cat";
    eprintln!("Prompt: \"{prompt}\", generating {num_images} image(s)");
    let tokenizer = tokenizers::Tokenizer::from_file(assets.join("tokenizer/tokenizer.json"))
        .map_err(|e| anyhow!("{e}"))?;
    let encode = |text: &str| -> Result<Vec<i64>> {
        let enc = tokenizer.encode(text, true).map_err(|e| anyhow!("{e}"))?;
        let mut ids: Vec<i64> = enc.get_ids().iter().map(|&id| id as i64).collect();
        ids.resize(77, tokenizer.token_to_id("<|endoftext|>").unwrap_or(49407) as i64);
        Ok(ids)
    };
    let cond_ids = encode(prompt)?;
    let uncond_ids = encode("")?;

    // Generate initial latent noise — one per image
    use rand::SeedableRng;
    use rand_distr::{Distribution, StandardNormal};
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let latent_size = 4 * 64 * 64;
    let initial_noise: Vec<f32> =
        (0..num_images * latent_size).map(|_| StandardNormal.sample(&mut rng)).collect();

    // Build scheduler
    let num_steps = 20;
    let scheduler = EulerScheduler::new(num_steps);
    let init_noise_sigma = scheduler.init_noise_sigma();
    eprintln!("  Scheduler: {num_steps} steps, init_sigma={init_noise_sigma:.4}");

    // Pick best available runtime
    let gpu = ["cuda", "metal", "default"]
        .iter()
        .find_map(|rt| tract::runtime_for_name(rt).ok())
        .unwrap();
    eprintln!("Using runtime: {gpu:?}");

    let b2 = 2 * num_images;

    eprintln!("Loading models...");
    let onnx = tract::onnx()?;
    let text_encoder = gpu.prepare(onnx.load(assets.join("text_encoder.onnx"))?.into_model()?)?;
    let unet = gpu.prepare(onnx.load(assets.join("unet.onnx"))?.into_model()?)?;
    let vae_decoder = gpu.prepare(onnx.load(assets.join("vae_decoder.onnx"))?.into_model()?)?;

    // --- Text encoding (batch=1 is fine, same embeddings for all images) ---
    eprintln!("Running text encoder...");
    let cond_arr = ndarray::Array2::from_shape_vec((1, 77), cond_ids)?;
    let uncond_arr = ndarray::Array2::from_shape_vec((1, 77), uncond_ids)?;
    let cond_emb = text_encoder.run([cond_arr])?;
    let uncond_emb = text_encoder.run([uncond_arr])?;
    let cond_slice = cond_emb[0].as_slice::<f32>()?; // (1, 77, 768)
    let uncond_slice = uncond_emb[0].as_slice::<f32>()?;

    // Build batched embeddings: [uncond*N, cond*N] -> (2N, 77, 768)
    let emb_size = 77 * 768;
    let mut emb_data = Vec::with_capacity(2 * num_images * emb_size);
    for _ in 0..num_images {
        emb_data.extend_from_slice(uncond_slice);
    }
    for _ in 0..num_images {
        emb_data.extend_from_slice(cond_slice);
    }
    let text_emb = tract_ndarray::ArrayD::from_shape_vec(vec![2 * num_images, 77, 768], emb_data)?;

    // --- Denoising loop (batched) ---
    eprintln!("Running denoising loop ({num_steps} steps, {num_images} images)...");
    let guidance_scale: f32 = 7.5;
    let batch_latent_size = num_images * latent_size;
    let mut latents: Vec<f32> = initial_noise.iter().map(|&x| x * init_noise_sigma).collect();

    for (i, &t) in scheduler.timesteps.iter().enumerate() {
        let scale = scheduler.scale_factor(i);

        // Build UNet input: [scaled_latents * N, scaled_latents * N] -> (2N, 4, 64, 64)
        let mut sample_data = Vec::with_capacity(2 * batch_latent_size);
        for &x in &latents {
            sample_data.push(x * scale);
        }
        for &x in &latents {
            sample_data.push(x * scale);
        }
        let sample =
            tract_ndarray::ArrayD::from_shape_vec(vec![2 * num_images, 4, 64, 64], sample_data)?;
        let timestep = tract_ndarray::Array1::from_vec(vec![t; b2]).into_dyn();

        let noise_pred =
            unet.run(vec![tensor(sample)?, tensor(timestep)?, tensor(text_emb.clone())?])?;
        let pred = noise_pred[0].as_slice::<f32>()?;

        // CFG: split uncond (first N) / cond (last N), combine
        let dt = scheduler.dt(i);
        for j in 0..batch_latent_size {
            let u = pred[j];
            let c = pred[batch_latent_size + j];
            let eps = u + guidance_scale * (c - u);
            latents[j] += eps * dt;
        }

        if i % 5 == 0 {
            let sigma = scheduler.sigmas[i];
            eprintln!("  Step {i}/{num_steps}, t={t}, sigma={sigma:.4}");
        }
    }

    // --- VAE decode (batched) ---
    eprintln!("Running VAE decoder...");
    let latent_scaled: Vec<f32> = latents.iter().map(|&x| x / VAE_SCALING_FACTOR).collect();
    let latent_arr =
        tract_ndarray::ArrayD::from_shape_vec(vec![num_images, 4, 64, 64], latent_scaled)?;
    let image_result = vae_decoder.run([latent_arr])?;
    let image_data = image_result[0].as_slice::<f32>()?;

    // Save each image as PNG (NCHW -> RGB pixels)
    let (h, w) = (512usize, 512usize);
    let img_size = 3 * h * w;
    for n in 0..num_images {
        let offset = n * img_size;
        let mut pixels = vec![0u8; h * w * 3];
        for y in 0..h {
            for x in 0..w {
                for ch in 0..3 {
                    let val = (image_data[offset + ch * h * w + y * w + x].clamp(-1.0, 1.0) + 1.0)
                        / 2.0
                        * 255.0;
                    pixels[(y * w + x) * 3 + ch] = val as u8;
                }
            }
        }
        let path = if num_images == 1 {
            assets.join("output.png")
        } else {
            assets.join(format!("output_{n}.png"))
        };
        image::save_buffer(&path, &pixels, w as u32, h as u32, image::ColorType::Rgb8)?;
        eprintln!("Saved {}", path.display());
    }

    Ok(())
}
