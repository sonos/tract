//! Runs a segmenter natively on the wgpu runtime: checks that every compute
//! node is on the GPU, compares the mask with the CPU's on a synthetic frame,
//! and times a run. The same model file and normalization the page uses.
//!
//! Usage: pixel-lab-native <model.onnx> <width> <height> [mean] [std]
//! where mean and std are comma-separated, one value per RGB channel
//! (default 0,0,0 and 1,1,1).

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use tract::prelude::*;

fn triple(arg: Option<String>, default: f32) -> Result<[f32; 3]> {
    let Some(arg) = arg else { return Ok([default; 3]) };
    let values = arg.split(',').map(str::parse).collect::<Result<Vec<f32>, _>>()?;
    values.try_into().map_err(|_| anyhow::anyhow!("Expected three comma-separated values"))
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args.next().context("Pass the ONNX model path")?;
    let width: usize = args.next().context("Pass the model width")?.parse()?;
    let height: usize = args.next().context("Pass the model height")?.parse()?;
    let mean = triple(args.next(), 0.0)?;
    let std = triple(args.next(), 1.0)?;

    let mut inference =
        tract::onnx()?.load_with_options(path, OnnxOptions::new().ignore_value_info(true))?;
    inference.set_input_fact(0, format!("1,3,{height},{width},f32").as_str())?;
    inference.set_output_fact(0, ())?;
    let model = inference.into_model()?;
    let gpu = tract::runtime_for_name("wgpu")?.prepare(model.clone())?;
    let cpu = model.into_runnable()?;

    let graph: serde_json::Value = serde_json::from_str(&gpu.cost_json()?)?;
    let mut compute = 0;
    for node in graph["nodes"].as_array().context("Missing model graph")? {
        let op = node["op_name"].as_str().context("Missing op name")?;
        if op.starts_with("Wgpu") || op.starts_with("Gpu") {
            compute += 1;
        } else {
            ensure!(
                matches!(op, "Const" | "Source" | "DeviceSyncToDevice" | "DeviceSyncToHost"),
                "CPU fallback: {op}"
            );
        }
    }
    ensure!(compute > 0, "No GPU compute nodes");

    let pixels = width * height;
    let mut rgb = vec![0f32; 3 * pixels];
    for i in 0..pixels {
        let (x, y) = ((i % width) as f32 / width as f32, (i / width) as f32 / height as f32);
        let value = [x, y, 0.5 + 0.5 * (7.0 * x * y).sin()];
        for c in 0..3 {
            rgb[c * pixels + i] = (value[c] - mean[c]) / std[c];
        }
    }
    let input = || Tensor::from_slice(&[1, 3, height, width], &rgb);
    let actual = gpu.run([input()?])?;
    let expected = cpu.run([input()?])?;
    let (actual, expected) = (actual[0].as_slice::<f32>()?, expected[0].as_slice::<f32>()?);
    ensure!(actual.len() == pixels && expected.len() == pixels, "Unexpected mask size");
    let mut max_error = 0f32;
    for (a, e) in actual.iter().zip(expected) {
        ensure!(a.is_finite() && e.is_finite(), "Non-finite mask");
        max_error = max_error.max((a - e).abs());
    }
    ensure!(max_error < 1e-3, "GPU and CPU masks differ by {max_error}");
    println!("PASS: {compute} GPU ops; {pixels} mask values; max GPU/CPU error {max_error:.3e}");

    for _ in 0..3 {
        gpu.run([input()?])?;
    }
    let mut times = (0..30)
        .map(|_| {
            let started = std::time::Instant::now();
            gpu.run([input()?])?;
            Ok(started.elapsed().as_secs_f64() * 1e3)
        })
        .collect::<Result<Vec<_>>>()?;
    times.sort_by(f64::total_cmp);
    println!(
        "Native GPU run with readback: median {:.2} ms (not browser FPS)",
        times[times.len() / 2]
    );
    Ok(())
}
