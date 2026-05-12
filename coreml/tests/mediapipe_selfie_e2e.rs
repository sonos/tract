//! Phase 3 third-canary probe: MediaPipe Selfie segmentation through `CoremlRuntime`.
//!
//! Fixed shape `[1,3,256,256] -> [1,1,256,256]`. Op-mix per ONNX inspection:
//!   54× Conv, 22× Relu, 14× Add, 11× HardSwish, 11× Sigmoid, 10× ReduceMean,
//!   10× Mul, 3× Concat, 3× Resize, 3× Shape, 3× Slice, 1× ConvTranspose
//!
//! New ops needed beyond MobileNet+SqueezeNet+Phase-3-D set: ConvTranspose,
//! Slice, dynamic-shape handling. Resize / Sigmoid / HardSwish / Concat /
//! Reduce now covered.
//!
//! The MediaPipe Selfie ONNX export carries `batch_size` as a symbolic dim
//! baked into conv output facts, which tract-onnx can't unify with an
//! override input fact alone. We work around this by overriding the input
//! fact AND running `analyse(true)` so tract propagates the concrete value.
//! If that still fails, we fall back to the `_fp16` variant.
//!
//! Run with:
//! ```sh
//! cargo test -p tract-coreml --test mediapipe_selfie_e2e -- --ignored --nocapture
//! ```

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use half::f16;

use tract_core::internal::*;
use tract_core::ops::cnn::Conv;
use tract_core::transform::{ModelTransform, get_transform};
use tract_onnx::prelude::*;

use tract_coreml::{CoremlOp, CoremlTransform};

const MODEL_PATHS: &[&str] = &[
    "~/coding/v7-webgl-relighting-worktree/models/onnx/mediapipe_selfie.onnx",
    "~/coding/v7-webgl-relighting-worktree/models/onnx/mediapipe_selfie_fp16.onnx",
];
const INPUT_SHAPE: [usize; 4] = [1, 3, 256, 256];

fn resolve_path(path: &str) -> PathBuf {
    if let Some(stripped) = path.strip_prefix("~/") {
        let home = std::env::var("HOME").expect("HOME unset");
        PathBuf::from(home).join(stripped)
    } else {
        PathBuf::from(path)
    }
}

fn load_model_f16() -> Result<TypedModel> {
    // Try each model path in order; first one that loads + types wins.
    let mut last_err: Option<anyhow::Error> = None;
    for raw_path in MODEL_PATHS {
        let path = resolve_path(raw_path);
        if !path.exists() {
            continue;
        }
        match try_load(&path) {
            Ok(m) => return Ok(m),
            Err(e) => {
                println!("[NOTE] {}: {e}", path.display());
                last_err = Some(e);
            }
        }
    }
    Err(last_err.unwrap_or_else(|| anyhow::anyhow!("no MediaPipe Selfie model found")))
}

fn try_load(path: &PathBuf) -> Result<TypedModel> {
    let mut inf = tract_onnx::onnx().model_for_path(path)?;
    inf.set_input_fact(
        0,
        InferenceFact::dt_shape(
            f32::datum_type(),
            tvec![
                INPUT_SHAPE[0] as i64,
                INPUT_SHAPE[1] as i64,
                INPUT_SHAPE[2] as i64,
                INPUT_SHAPE[3] as i64,
            ],
        ),
    )?;
    inf.analyse(true)?;
    let model = inf.into_typed()?;
    println!("loaded {}: {} nodes (batch_size bound to 1)", path.display(), model.nodes.len());

    let model = model.into_decluttered()?;
    println!("after into_decluttered: {} nodes", model.nodes.len());
    let mut model = model;
    let f16 = get_transform("f32_to_f16")?
        .ok_or_else(|| anyhow::anyhow!("f32_to_f16 transform not registered"))?;
    f16.transform(&mut model)?;
    println!("after f32_to_f16: {} nodes", model.nodes.len());
    Ok(model)
}

fn make_input() -> Result<Tensor> {
    let n: usize = INPUT_SHAPE.iter().product();
    let data: Vec<f16> = (0..n).map(|i| f16::from_f32(((i as f32) * 0.001).sin() * 0.5)).collect();
    Tensor::from_shape::<f16>(&INPUT_SHAPE, &data)
}

#[test]
#[ignore = "loads external mediapipe_selfie.onnx; --ignored --nocapture"]
fn mediapipe_selfie_coverage_probe() -> Result<()> {
    let cpu_model = load_model_f16()?;

    let mut op_hist: std::collections::BTreeMap<String, usize> = Default::default();
    for n in &cpu_model.nodes {
        *op_hist.entry(n.op.name().to_string()).or_default() += 1;
    }
    println!("\n[op-type histogram, post-declutter+f16, {} nodes]", cpu_model.nodes.len());
    for (op, c) in &op_hist {
        println!("  {c:4}× {op}");
    }

    use tract_coreml::ops::conv::{ConvAnalysis, analyse_conv};
    let conv_ids: Vec<usize> =
        cpu_model.nodes.iter().filter(|n| n.op_as::<Conv>().is_some()).map(|n| n.id).collect();
    let mut t = 0;
    let mut skip_counts: std::collections::BTreeMap<String, usize> = Default::default();
    for id in &conv_ids {
        match analyse_conv(&cpu_model, &cpu_model.nodes[*id])? {
            ConvAnalysis::Translatable(_) => t += 1,
            ConvAnalysis::Skip(r) => {
                let bucket = r.split(' ').take(4).collect::<Vec<_>>().join(" ");
                *skip_counts.entry(bucket).or_default() += 1;
            }
        }
    }
    println!("\n[per-Conv] {t}/{} translatable", conv_ids.len());
    for (r, c) in &skip_counts {
        println!("  {c}× {r}");
    }

    let mut coreml_model = cpu_model.clone();
    let t0 = Instant::now();
    match CoremlTransform::default().transform(&mut coreml_model) {
        Ok(()) => println!("\nCoremlTransform: {:?}", t0.elapsed()),
        Err(e) => {
            println!("\nCoremlTransform FAILED: {e}");
            return Ok(());
        }
    }

    let coreml_count =
        coreml_model.nodes.iter().filter(|n| n.op_as::<CoremlOp>().is_some()).count();
    let convs_after = coreml_model.nodes.iter().filter(|n| n.op_as::<Conv>().is_some()).count();
    println!("=== after CoremlTransform ===");
    println!("  total nodes: {}", coreml_model.nodes.len());
    println!("  Conv left:   {convs_after}");
    println!("  CoremlOps:   {coreml_count}");

    let mut after_hist: std::collections::BTreeMap<String, usize> = Default::default();
    for n in &coreml_model.nodes {
        *after_hist.entry(n.op.name().to_string()).or_default() += 1;
    }
    println!("\n[post-transform op histogram]");
    for (op, c) in &after_hist {
        println!("  {c:4}× {op}");
    }

    let input = make_input()?;
    let cpu_runnable = cpu_model.into_runnable()?;
    let coreml_runnable = match coreml_model.into_runnable() {
        Ok(r) => r,
        Err(e) => {
            println!("\n[NOTE] coreml_model.into_runnable() failed: {e}");
            let _ = cpu_runnable.run(tvec![input.into()])?;
            return Ok(());
        }
    };

    let cpu_out = {
        let t0 = Instant::now();
        let out = cpu_runnable.run(tvec![input.clone().into()])?;
        println!("CPU first run: {:?}", t0.elapsed());
        out
    };
    let coreml_out = {
        let t0 = Instant::now();
        let out = coreml_runnable.run(tvec![input.clone().into()])?;
        println!("CoreML first run: {:?}", t0.elapsed());
        out
    };
    println!("CPU shape: {:?}", cpu_out[0].shape());
    println!("CoreML shape: {:?}", coreml_out[0].shape());

    let c_slice = unsafe { cpu_out[0].as_slice_unchecked::<f16>() };
    let k_slice = unsafe { coreml_out[0].as_slice_unchecked::<f16>() };
    let mut max_abs = 0.0f32;
    let mut sq_diff = 0.0f64;
    let mut sq_ref = 0.0f64;
    for (cf, kf) in c_slice.iter().zip(k_slice.iter()) {
        let d = (cf.to_f32() - kf.to_f32()).abs();
        if d > max_abs {
            max_abs = d;
        }
        sq_diff += (d as f64) * (d as f64);
        sq_ref += (cf.to_f32() as f64).powi(2);
    }
    let rel_l2 = (sq_diff / sq_ref).sqrt();
    println!("\n[mediapipe selfie] max_abs={max_abs:.6}  rel_l2={rel_l2:.6e}");

    Ok(())
}

#[test]
#[ignore = "loads external mediapipe_selfie.onnx; --ignored --nocapture"]
fn mediapipe_selfie_perf_cpu_vs_coreml() -> Result<()> {
    const ITERATIONS: usize = 200;
    const WARMUP: usize = 5;

    let cpu_model = load_model_f16()?;
    let mut coreml_model = cpu_model.clone();
    CoremlTransform::default().transform(&mut coreml_model)?;

    let cpu_runnable = cpu_model.into_runnable()?;
    let coreml_runnable = coreml_model.into_runnable()?;

    let input = make_input()?;

    for _ in 0..WARMUP {
        let _ = cpu_runnable.run(tvec![input.clone().into()])?;
        let _ = coreml_runnable.run(tvec![input.clone().into()])?;
    }

    let t0 = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = cpu_runnable.run(tvec![input.clone().into()])?;
    }
    let cpu_per = t0.elapsed() / ITERATIONS as u32;

    let t0 = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = coreml_runnable.run(tvec![input.clone().into()])?;
    }
    let coreml_per = t0.elapsed() / ITERATIONS as u32;

    let speedup = cpu_per.as_secs_f64() / coreml_per.as_secs_f64();
    let coreml_ms = coreml_per.as_secs_f64() * 1000.0;
    let rtf_30fps = coreml_ms / 33.333;
    let rtf_60fps = coreml_ms / 16.667;
    println!("\n=== MediaPipe Selfie perf (n={ITERATIONS}, warmup={WARMUP}) ===");
    println!("  CPU mean:    {cpu_per:?}");
    println!("  CoreML mean: {coreml_per:?}");
    println!("  Speedup:     {speedup:.2}x");
    println!("  RTF (30fps): {rtf_30fps:.4}");
    println!("  RTF (60fps): {rtf_60fps:.4}");
    Ok(())
}
