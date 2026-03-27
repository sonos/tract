use anyhow::*;
use tract::prelude::*;

/// Read a flat binary file of f32 values and reshape.
fn read_f32(
    path: impl AsRef<std::path::Path>,
    shape: &[usize],
) -> Result<tract_ndarray::ArrayD<f32>> {
    let bytes = std::fs::read(path)?;
    let data: Vec<f32> =
        bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    Ok(tract_ndarray::ArrayD::from_shape_vec(shape.to_vec(), data)?)
}

/// Read a flat binary file of i64 values and reshape.
fn read_i64(
    path: impl AsRef<std::path::Path>,
    shape: &[usize],
) -> Result<tract_ndarray::ArrayD<i64>> {
    let bytes = std::fs::read(path)?;
    let data: Vec<i64> =
        bytes.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect();
    Ok(tract_ndarray::ArrayD::from_shape_vec(shape.to_vec(), data)?)
}

fn main() -> Result<()> {
    let assets = std::path::Path::new("assets");

    // Load reference data (pre-computed by reference.py)
    eprintln!("Loading reference data...");
    let input_ids = read_i64(assets.join("input_ids.bin"), &[1, 77])?;
    let uncond_input_ids = read_i64(assets.join("uncond_input_ids.bin"), &[1, 77])?;
    let initial_latent = read_f32(assets.join("initial_latent.bin"), &[1, 4, 64, 64])?;
    let ts_bytes = std::fs::read(assets.join("timesteps.bin"))?;
    let timesteps: Vec<i64> =
        ts_bytes.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect();

    // Read scalar params
    let params = std::fs::read_to_string(assets.join("params.txt"))?;
    let mut vae_scaling_factor: f32 = 0.0;
    let mut init_noise_sigma: f32 = 0.0;
    for line in params.lines() {
        if let Some(v) = line.strip_prefix("vae_scaling_factor=") {
            vae_scaling_factor = v.parse()?;
        }
        if let Some(v) = line.strip_prefix("init_noise_sigma=") {
            init_noise_sigma = v.parse()?;
        }
    }

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
    eprintln!("  Text embeddings shape: {:?}", text_embeddings.shape());

    // Split text embeddings for separate uncond/cond UNet runs
    let uncond_emb_arr =
        text_embeddings.slice(tract_ndarray::s![0..1, .., ..]).to_owned().into_dyn();
    let cond_emb_arr = text_embeddings.slice(tract_ndarray::s![1..2, .., ..]).to_owned().into_dyn();

    // --- Denoising loop ---
    let n_steps = timesteps.len();
    eprintln!("Running denoising loop ({n_steps} steps)...");
    let guidance_scale: f32 = 7.5;
    let mut latent = initial_latent.mapv(|x| x * init_noise_sigma);

    for (i, &t) in timesteps.iter().enumerate() {
        let timestep = tract_ndarray::arr1(&[t]).into_dyn();

        // Run UNet twice: unconditional and conditional
        let uncond_pred = unet.run(vec![
            tensor(latent.clone())?,
            tensor(timestep.clone())?,
            tensor(uncond_emb_arr.clone())?,
        ])?;
        let cond_pred = unet.run(vec![
            tensor(latent.clone())?,
            tensor(timestep)?,
            tensor(cond_emb_arr.clone())?,
        ])?;

        let u = uncond_pred[0].as_slice::<f32>()?;
        let c = cond_pred[0].as_slice::<f32>()?;

        // Classifier-free guidance
        let guided: Vec<f32> =
            u.iter().zip(c).map(|(&u, &c)| u + guidance_scale * (c - u)).collect();

        // TODO: proper scheduler step (Euler/PNDM)
        latent = tract_ndarray::ArrayD::from_shape_vec(vec![1, 4, 64, 64], guided)?;

        if i % 5 == 0 {
            eprintln!("  Step {i}/{n_steps}, t={t}");
        }
    }

    // --- VAE decode ---
    eprintln!("Running VAE decoder...");
    let latent_scaled = latent.mapv(|x| x / vae_scaling_factor);
    let image_result = vae_decoder.run([latent_scaled])?;
    let image_data = image_result[0].as_slice::<f32>()?;
    eprintln!("  Image: {} floats", image_data.len());

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
