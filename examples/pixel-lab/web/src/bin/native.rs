//! Runs a segmenter natively on the wgpu runtime: checks that every compute
//! node is on the GPU, compares the mask with the CPU's on a synthetic frame,
//! and times a run. The same model file and normalization the page uses.
//!
//! Usage: pixel-lab-native <model.onnx> <width> <height> [mean] [std]
//! where mean and std are comma-separated, one value per RGB channel
//! (default 0,0,0 and 1,1,1).
//!
//! RVM: pixel-lab-native --rvm <model.onnx> <width> <height> <ratio>
//! `ratio` is the downsample scale for that camera size. RGB stays in [0, 1].

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use tract::prelude::*;

fn triple(arg: Option<String>, default: f32) -> Result<[f32; 3]> {
    let Some(arg) = arg else { return Ok([default; 3]) };
    let values = arg.split(',').map(str::parse).collect::<Result<Vec<f32>, _>>()?;
    values.try_into().map_err(|_| anyhow::anyhow!("Expected three comma-separated values"))
}

fn rvm_states(width: usize, height: usize, ratio: f32) -> Result<[[usize; 4]; 4]> {
    ensure!(ratio > 0.0 && ratio <= 1.0, "RVM scale must be in (0, 1]");
    let w = (width as f32 * ratio).round() as usize;
    let h = (height as f32 * ratio).round() as usize;
    ensure!(w > 0 && h > 0, "RVM scale collapses the camera frame");
    let spec = [(16usize, 2usize), (20, 4), (40, 8), (64, 16)];
    Ok(spec.map(|(channels, stride)| [1, channels, h.div_ceil(stride), w.div_ceil(stride)]))
}

fn run_rvm(mut args: impl Iterator<Item = String>) -> Result<()> {
    let path = args.next().context("Pass the RVM ONNX path")?;
    let width: usize = args.next().context("Pass the camera width")?.parse()?;
    let height: usize = args.next().context("Pass the camera height")?.parse()?;
    let ratio: f32 = args.next().context("Pass the RVM scale")?.parse()?;
    let shapes = rvm_states(width, height, ratio)?;
    let options = OnnxOptions::new().ignore_value_info(true).constant_input(
        "downsample_ratio",
        DatumType::F32,
        &[1],
        &[f64::from(ratio)],
    );
    let mut inference = tract::onnx()?.load_with_options(path, options)?;
    let expected_in = ["src", "r1i", "r2i", "r3i", "r4i"];
    ensure!(inference.input_count()? == expected_in.len(), "Unexpected RVM inputs");
    for (i, name) in expected_in.iter().enumerate() {
        ensure!(inference.input_name(i)? == *name, "RVM input {i} is not {name}");
    }
    let declared = inference.input_fact(0)?.to_string();
    ensure!(
        declared.ends_with("f32") || declared.ends_with("f32?"),
        "RVM src is {declared}, expected f32"
    );
    inference.set_input_fact(0, format!("1,3,{height},{width},f32").as_str())?;
    for (i, shape) in shapes.iter().enumerate() {
        inference.set_input_fact(
            i + 1,
            format!("{},{},{},{},f32", shape[0], shape[1], shape[2], shape[3]).as_str(),
        )?;
    }
    let output_count = inference.output_count()?;
    ensure!(output_count == 6, "RVM should have 6 outputs, found {output_count}");
    for i in 0..output_count {
        inference.set_output_fact(i, ())?;
    }
    let model = inference.into_model()?;
    let mut expected =
        vec![format!("1,3,{height},{width},f32"), format!("1,1,{height},{width},f32")];
    for shape in &shapes {
        expected.push(format!("{},{},{},{},f32", shape[0], shape[1], shape[2], shape[3]));
    }
    for (i, shape) in expected.iter().enumerate() {
        let actual = model.output_fact(i)?.to_string();
        ensure!(actual == *shape, "RVM output {i} is {actual}, expected {shape}");
    }
    let gpu = tract::runtime_for_name("wgpu")?.prepare(model.clone())?;
    let cpu = model.into_runnable()?;
    let spec = gpu.cost_json()?;
    let f16_nodes = spec.matches("\"F16\"").count();
    let graph: serde_json::Value = serde_json::from_str(&spec)?;
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
        rgb[i] = value[0];
        rgb[pixels + i] = value[1];
        rgb[2 * pixels + i] = value[2];
    }
    let image = || Tensor::from_slice(&[1, 3, height, width], &rgb);
    let zeros = |shape: &[usize; 4]| Tensor::from_slice(shape, &vec![0f32; shape.iter().product()]);
    let mut gpu_state = shapes.iter().map(zeros).collect::<Result<Vec<_>>>()?;
    let cpu_state = shapes.iter().map(zeros).collect::<Result<Vec<_>>>()?;
    let inputs = |state: &[Tensor]| {
        let mut all = vec![image()?];
        all.extend(state.iter().cloned());
        Ok::<_, anyhow::Error>(all)
    };
    let gpu_out = gpu.run(inputs(&gpu_state)?)?;
    let cpu_out = cpu.run(inputs(&cpu_state)?)?;
    let (actual, expected) = (gpu_out[1].as_slice::<f32>()?, cpu_out[1].as_slice::<f32>()?);
    ensure!(actual.len() == pixels && expected.len() == pixels, "Unexpected alpha size");
    let mut max_error = 0f32;
    for (a, e) in actual.iter().zip(expected) {
        ensure!(a.is_finite() && e.is_finite(), "Non-finite alpha");
        max_error = max_error.max((a - e).abs());
    }
    ensure!(max_error < 1e-3, "GPU and CPU alpha differ by {max_error}");
    gpu_state = gpu_out[2..].to_vec();
    let again = gpu.run(inputs(&gpu_state)?)?;
    ensure!(again[1].as_slice::<f32>()?.iter().all(|v| v.is_finite()), "Non-finite second frame");
    println!(
        "PASS: RVM scale {ratio}, network {}x{}, {compute} GPU ops, {f16_nodes} f16 facts, max GPU/CPU error {max_error:.3e}",
        (width as f32 * ratio).round() as usize,
        (height as f32 * ratio).round() as usize
    );
    Ok(())
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args.next().context("Pass the ONNX model path")?;
    if path == "--rvm" {
        return run_rvm(args);
    }
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
