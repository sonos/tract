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

        // cumulative product → sigmas
        let mut alphas_cumprod = alphas.clone();
        for i in 1..num_train {
            alphas_cumprod[i] *= alphas_cumprod[i - 1];
        }
        let all_sigmas = alphas_cumprod.mapv(|a| ((1.0 - a) / a).sqrt());

        // Timesteps: linspace num_train-1 → 0
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

    fn scale_model_input(&self, latent: &[f32], step: usize) -> Vec<f32> {
        let sigma = self.sigmas[step];
        let scale = 1.0 / (sigma * sigma + 1.0).sqrt();
        latent.iter().map(|&x| x * scale).collect()
    }

    fn step(&self, latent: &[f32], noise_pred: &[f32], step: usize) -> Vec<f32> {
        let sigma = self.sigmas[step];
        let sigma_next = self.sigmas[step + 1];
        let dt = sigma_next - sigma;
        latent.iter().zip(noise_pred).map(|(&x, &eps)| x + eps * dt).collect()
    }
}

/// SD 1.5 VAE scaling factor
const VAE_SCALING_FACTOR: f32 = 0.18215;

fn main() -> Result<()> {
    let assets = std::path::Path::new("assets");

    // Load tokenized prompt from npz (only tokens + initial latent needed)
    eprintln!("Loading pipeline data...");
    let npz_bytes = std::fs::read(assets.join("pipeline.npz"))?;
    let mut npz = ndarray_npy::NpzReader::new(std::io::Cursor::new(npz_bytes))?;
    let input_ids: ndarray::Array2<i64> = npz.by_name("input_ids")?;
    let uncond_input_ids: ndarray::Array2<i64> = npz.by_name("uncond_input_ids")?;

    // Generate initial latent noise
    use rand::SeedableRng;
    use rand_distr::{Distribution, StandardNormal};
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let initial_latent =
        ndarray::Array4::<f32>::from_shape_fn((1, 4, 64, 64), |_| StandardNormal.sample(&mut rng));

    // Build scheduler from scratch
    let num_steps = 20;
    let scheduler = EulerScheduler::new(num_steps);
    let init_noise_sigma = scheduler.init_noise_sigma();
    eprintln!("  Scheduler: {num_steps} steps, init_sigma={init_noise_sigma:.4}");

    // Pick best available runtime (CUDA > Metal > CPU)
    let gpu = ["cuda", "metal", "default"]
        .iter()
        .find_map(|rt| tract::runtime_for_name(rt).ok())
        .unwrap();
    eprintln!("Using runtime: {gpu:?}");

    eprintln!("Loading models...");
    let onnx = tract::onnx()?;
    let text_encoder = gpu.prepare(onnx.load(assets.join("text_encoder.onnx"))?.into_model()?)?;
    let unet = gpu.prepare(onnx.load(assets.join("unet.onnx"))?.into_model()?)?;
    let vae_decoder = gpu.prepare(onnx.load(assets.join("vae_decoder.onnx"))?.into_model()?)?;

    // --- Text encoding ---
    eprintln!("Running text encoder...");
    let cond_emb = text_encoder.run([input_ids])?;
    let uncond_emb = text_encoder.run([uncond_input_ids])?;

    let cond = cond_emb[0].view::<f32>()?;
    let uncond = uncond_emb[0].view::<f32>()?;
    let text_embeddings =
        tract_ndarray::concatenate(tract_ndarray::Axis(0), &[uncond.view(), cond.view()])?;
    eprintln!("  Text embeddings: {:?}", text_embeddings.shape());

    // Split for separate uncond/cond UNet runs (batch=1 model)
    let uncond_emb_arr =
        text_embeddings.slice(tract_ndarray::s![0..1, .., ..]).to_owned().into_dyn();
    let cond_emb_arr = text_embeddings.slice(tract_ndarray::s![1..2, .., ..]).to_owned().into_dyn();

    // --- Denoising loop ---
    eprintln!("Running denoising loop ({num_steps} steps, Euler scheduler)...");
    let guidance_scale: f32 = 7.5;
    let mut latent: Vec<f32> =
        initial_latent.as_slice().unwrap().iter().map(|&x| x * init_noise_sigma).collect();

    for (i, &t) in scheduler.timesteps.iter().enumerate() {
        let scaled = scheduler.scale_model_input(&latent, i);
        let scaled_arr = tract_ndarray::ArrayD::from_shape_vec(vec![1, 4, 64, 64], scaled)?;
        let timestep = tract_ndarray::arr1(&[t]).into_dyn();

        // Run UNet twice: unconditional and conditional
        let uncond_pred = unet.run(vec![
            tensor(scaled_arr.clone())?,
            tensor(timestep.clone())?,
            tensor(uncond_emb_arr.clone())?,
        ])?;
        let cond_pred =
            unet.run(vec![tensor(scaled_arr)?, tensor(timestep)?, tensor(cond_emb_arr.clone())?])?;

        let u = uncond_pred[0].as_slice::<f32>()?;
        let c = cond_pred[0].as_slice::<f32>()?;

        // Classifier-free guidance
        let noise_pred: Vec<f32> =
            u.iter().zip(c).map(|(&u, &c)| u + guidance_scale * (c - u)).collect();

        // Euler step
        latent = scheduler.step(&latent, &noise_pred, i);

        if i % 5 == 0 {
            let sigma = scheduler.sigmas[i];
            eprintln!("  Step {i}/{num_steps}, t={t}, sigma={sigma:.4}");
        }
    }

    // --- VAE decode ---
    eprintln!("Running VAE decoder...");
    let latent_scaled: Vec<f32> = latent.iter().map(|&x| x / VAE_SCALING_FACTOR).collect();
    let latent_arr = tract_ndarray::ArrayD::from_shape_vec(vec![1, 4, 64, 64], latent_scaled)?;
    let image_result = vae_decoder.run([latent_arr])?;
    let image_data = image_result[0].as_slice::<f32>()?;

    // Save as PNG (NCHW -> RGB pixels)
    let (h, w) = (512usize, 512usize);
    let mut pixels = vec![0u8; h * w * 3];
    for y in 0..h {
        for x in 0..w {
            for ch in 0..3 {
                let val = (image_data[ch * h * w + y * w + x].clamp(-1.0, 1.0) + 1.0) / 2.0 * 255.0;
                pixels[(y * w + x) * 3 + ch] = val as u8;
            }
        }
    }
    image::save_buffer(
        assets.join("output.png"),
        &pixels,
        w as u32,
        h as u32,
        image::ColorType::Rgb8,
    )?;
    eprintln!("Saved output to assets/output.png");

    Ok(())
}
