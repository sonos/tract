use anyhow::*;
use tract::prelude::*;

fn main() -> Result<()> {
    let assets = std::path::Path::new("assets");

    // Load pipeline data from npz
    eprintln!("Loading pipeline data...");
    let npz_bytes = std::fs::read(assets.join("pipeline.npz"))?;
    let mut npz = ndarray_npy::NpzReader::new(std::io::Cursor::new(npz_bytes))?;
    let input_ids: ndarray::Array2<i64> = npz.by_name("input_ids")?;
    let uncond_input_ids: ndarray::Array2<i64> = npz.by_name("uncond_input_ids")?;
    let initial_latent: ndarray::Array4<f32> = npz.by_name("initial_latent")?;
    let timesteps: ndarray::Array1<i64> = npz.by_name("timesteps")?;
    let sigmas: ndarray::Array1<f32> = npz.by_name("sigmas")?;
    let vae_sf: ndarray::Array1<f32> = npz.by_name("vae_scaling_factor")?;
    let vae_scaling_factor = vae_sf[0];
    let ins: ndarray::Array1<f32> = npz.by_name("init_noise_sigma")?;
    let init_noise_sigma = ins[0];

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

    // --- Denoising loop (Euler discrete scheduler) ---
    let n_steps = timesteps.len();
    eprintln!("Running denoising loop ({n_steps} steps, Euler scheduler)...");
    let guidance_scale: f32 = 7.5;
    let mut latent = initial_latent.mapv(|x| x * init_noise_sigma).into_dyn();

    for (i, &t) in timesteps.iter().enumerate() {
        let sigma = sigmas[i];
        let sigma_next = sigmas[i + 1];

        // Euler scale_model_input: sample / sqrt(sigma^2 + 1)
        let scale = 1.0 / (sigma * sigma + 1.0).sqrt();
        let scaled_latent = latent.mapv(|x| x * scale);

        let timestep = tract_ndarray::arr1(&[t]).into_dyn();

        // Run UNet twice: unconditional and conditional
        let uncond_pred = unet.run(vec![
            tensor(scaled_latent.clone())?,
            tensor(timestep.clone())?,
            tensor(uncond_emb_arr.clone())?,
        ])?;
        let cond_pred = unet.run(vec![
            tensor(scaled_latent)?,
            tensor(timestep)?,
            tensor(cond_emb_arr.clone())?,
        ])?;

        let u = uncond_pred[0].as_slice::<f32>()?;
        let c = cond_pred[0].as_slice::<f32>()?;

        // Classifier-free guidance
        let noise_pred: Vec<f32> =
            u.iter().zip(c).map(|(&u, &c)| u + guidance_scale * (c - u)).collect();

        // Euler step: latent = latent + noise_pred * (sigma_next - sigma)
        let dt = sigma_next - sigma;
        let new_latent: Vec<f32> = latent
            .as_slice()
            .unwrap()
            .iter()
            .zip(&noise_pred)
            .map(|(&x, &eps)| x + eps * dt)
            .collect();
        latent = tract_ndarray::ArrayD::from_shape_vec(vec![1, 4, 64, 64], new_latent)?;

        if i % 5 == 0 {
            eprintln!("  Step {i}/{n_steps}, t={t}, sigma={sigma:.4}");
        }
    }

    // --- VAE decode ---
    eprintln!("Running VAE decoder...");
    let latent_scaled = latent.mapv(|x| x / vae_scaling_factor);
    let image_result = vae_decoder.run([latent_scaled])?;
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
