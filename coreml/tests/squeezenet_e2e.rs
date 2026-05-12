//! Phase 3 second-canary: SqueezeNet ONNX through `CoremlRuntime`.
//!
//! Same shape contract as MobileNet (`[1,3,224,224] -> [1,1000]`) but exercises
//! a different op mix per the ONNX inspection:
//!   26× Conv, 26× Relu, 8× Concat, 3× MaxPool, 1× GlobalAveragePool, 1× Flatten
//!
//! Concat + MaxPool + GlobalAveragePool are the new translators needed beyond
//! the MobileNet set; this probe shows the per-op coverage + skip-reason
//! histogram to drive the translator work.
//!
//! `#[ignore]` by default; load model from outside the workspace. Run with:
//!
//! ```sh
//! cargo test -p tract-coreml --test squeezenet_e2e -- --ignored --nocapture
//! ```

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use half::f16;

use objc2_core_ml::MLComputeUnits;

use tract_core::internal::*;
use tract_core::ops::cnn::Conv;
use tract_core::transform::{ModelTransform, get_transform};
use tract_onnx::prelude::InferenceModelExt;

use tract_coreml::{CoremlOp, CoremlTransform};

const MODEL_PATH: &str = "~/coding/dfn3-wasm-relaxed-simd/models/squeezenet.onnx";
const INPUT_SHAPE: [usize; 4] = [1, 3, 224, 224];

fn resolve_path(path: &str) -> PathBuf {
    if let Some(stripped) = path.strip_prefix("~/") {
        let home = std::env::var("HOME").expect("HOME unset");
        PathBuf::from(home).join(stripped)
    } else {
        PathBuf::from(path)
    }
}

fn load_squeezenet_f16() -> Result<TypedModel> {
    let path = resolve_path(MODEL_PATH);
    let model = tract_onnx::onnx().model_for_path(&path)?.into_typed()?;
    println!("loaded {}: {} nodes", path.display(), model.nodes.len());

    let model = model.into_decluttered()?;
    println!("after into_decluttered: {} nodes", model.nodes.len());

    let mut model = model;
    let f16_xform = get_transform("f32_to_f16")?
        .ok_or_else(|| anyhow::anyhow!("f32_to_f16 transform not registered"))?;
    f16_xform.transform(&mut model)?;
    println!("after f32_to_f16: {} nodes", model.nodes.len());

    Ok(model)
}

fn make_input() -> Result<Tensor> {
    let n: usize = INPUT_SHAPE.iter().product();
    let data: Vec<f16> = (0..n).map(|i| f16::from_f32(((i as f32) * 0.001).sin() * 0.5)).collect();
    Tensor::from_shape::<f16>(&INPUT_SHAPE, &data)
}

#[test]
#[ignore = "loads external squeezenet.onnx; run with --ignored --nocapture"]
fn squeezenet_coverage_probe() -> Result<()> {
    let cpu_model = load_squeezenet_f16()?;

    // Op-type histogram, post-declutter+f16. Tells us which ops sit between
    // Convs (= what fusion needs to also translate to grow MLPackages past
    // per-Conv granularity).
    let mut op_hist: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    for n in &cpu_model.nodes {
        let key = n.op.name().to_string();
        *op_hist.entry(key).or_default() += 1;
    }
    println!("\n[op-type histogram, post-declutter+f16, {} nodes]", cpu_model.nodes.len());
    for (op, count) in &op_hist {
        println!("  {count:4}× {op}");
    }

    // Per-Conv eligibility breakdown.
    use tract_coreml::ops::conv::{ConvAnalysis, analyse_conv};
    let conv_node_ids: Vec<usize> =
        cpu_model.nodes.iter().filter(|n| n.op_as::<Conv>().is_some()).map(|n| n.id).collect();
    println!("\n[per-Conv eligibility] {} Conv nodes:", conv_node_ids.len());
    let mut translatable = 0;
    let mut skip_counts: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();
    for id in &conv_node_ids {
        let n = &cpu_model.nodes[*id];
        match analyse_conv(&cpu_model, n)? {
            ConvAnalysis::Translatable(_) => {
                translatable += 1;
            }
            ConvAnalysis::Skip(reason) => {
                let bucket = reason.split(' ').take(3).collect::<Vec<_>>().join(" ");
                *skip_counts.entry(bucket).or_default() += 1;
            }
        }
    }
    println!("  → {translatable}/{} translatable", conv_node_ids.len());
    if !skip_counts.is_empty() {
        println!("[Conv skip-reason histogram]");
        for (reason, count) in &skip_counts {
            println!("  {count:3}× {reason}…");
        }
    }

    // Apply CoremlTransform.
    let mut coreml_model = cpu_model.clone();
    let t0 = Instant::now();
    CoremlTransform::default().transform(&mut coreml_model)?;
    println!("\nCoremlTransform: {:?}", t0.elapsed());

    let coreml_count =
        coreml_model.nodes.iter().filter(|n| n.op_as::<CoremlOp>().is_some()).count();
    let convs_after = coreml_model.nodes.iter().filter(|n| n.op_as::<Conv>().is_some()).count();
    println!("=== after CoremlTransform ===");
    println!("  total nodes:        {}", coreml_model.nodes.len());
    println!("  Conv nodes left:    {convs_after}  (CPU fallback)");
    println!("  CoremlOp nodes:     {coreml_count}");

    // Op-type histogram of remaining (= what's still on CPU).
    let mut after_hist: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();
    for n in &coreml_model.nodes {
        let key = n.op.name().to_string();
        *after_hist.entry(key).or_default() += 1;
    }
    println!("\n[post-transform op histogram]");
    for (op, count) in &after_hist {
        println!("  {count:4}× {op}");
    }

    // Try to run end-to-end; if any op isn't supported the runnable will fail.
    // Either way, we get useful signal.
    let input = make_input()?;
    let cpu_runnable = cpu_model.into_runnable()?;
    let coreml_runnable = match coreml_model.into_runnable() {
        Ok(r) => r,
        Err(e) => {
            println!("\n[NOTE] coreml_model.into_runnable() failed: {e}");
            println!(
                "[NOTE] this typically means a CPU-fallback op needs work; \
                     CPU-only path still validated below"
            );
            let cpu_out = cpu_runnable.run(tvec![input.into()])?;
            println!("CPU first run output shape: {:?}", cpu_out[0].shape());
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

    assert_eq!(cpu_out.len(), 1);
    assert_eq!(coreml_out.len(), 1);
    let c = &cpu_out[0];
    let k = &coreml_out[0];
    println!("CPU    shape: {:?}", c.shape());
    println!("CoreML shape: {:?}", k.shape());
    assert_eq!(c.shape(), k.shape());

    let c_slice = unsafe { c.as_slice_unchecked::<f16>() };
    let k_slice = unsafe { k.as_slice_unchecked::<f16>() };

    let mut max_abs = 0.0f32;
    let mut sum_sq_diff = 0.0f64;
    let mut sum_sq_ref = 0.0f64;
    for (cf, kf) in c_slice.iter().zip(k_slice.iter()) {
        let cv = cf.to_f32();
        let kv = kf.to_f32();
        let d = (cv - kv).abs();
        if d > max_abs {
            max_abs = d;
        }
        sum_sq_diff += (d as f64) * (d as f64);
        sum_sq_ref += (cv as f64) * (cv as f64);
    }
    let rel_l2 = (sum_sq_diff / sum_sq_ref).sqrt();
    println!("\n[squeezenet cpu vs coreml] max_abs_diff = {max_abs:.6}  rel_l2 = {rel_l2:.6e}");

    let top1 = |slice: &[f16]| -> (usize, f32) {
        slice
            .iter()
            .enumerate()
            .map(|(i, v)| (i, v.to_f32()))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
    };
    let cpu_top1 = top1(c_slice);
    let coreml_top1 = top1(k_slice);
    println!("CPU    top-1: class {} (logit {:.4})", cpu_top1.0, cpu_top1.1);
    println!("CoreML top-1: class {} (logit {:.4})", coreml_top1.0, coreml_top1.1);

    Ok(())
}

#[test]
#[ignore = "loads external squeezenet.onnx; run after coverage_probe; --ignored --nocapture"]
fn squeezenet_perf_cpu_vs_coreml() -> Result<()> {
    const ITERATIONS: usize = 200;
    const WARMUP: usize = 5;

    let cpu_model = load_squeezenet_f16()?;
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

    println!("\n=== SqueezeNet perf (n={ITERATIONS}, warmup={WARMUP}) ===");
    println!("  CPU mean:    {cpu_per:?}");
    println!("  CoreML mean: {coreml_per:?}");
    println!("  Speedup:     {speedup:.2}x");
    println!("  RTF (30fps): {rtf_30fps:.4}");
    println!("  RTF (60fps): {rtf_60fps:.4}");

    Ok(())
}

/// Sustained-loop benchmark for `powermetrics` ANE measurement on SqueezeNet.
/// Runs the CoreML-dispatched model for ~30s under CPUAndNeuralEngine so a
/// separate `sudo powermetrics -A --show-extra-power-info` capture can record
/// ANE Power.
#[test]
#[ignore = "loads external squeezenet.onnx; long sustained run; pair with sudo powermetrics"]
fn squeezenet_sustained_for_powermetrics() -> Result<()> {
    let mut model = load_squeezenet_f16()?;
    let xform = CoremlTransform { compute_units: MLComputeUnits::CPUAndNeuralEngine };
    xform.transform(&mut model)?;

    let coreml_count = model.nodes.iter().filter(|n| n.op_as::<CoremlOp>().is_some()).count();
    println!("Compiled with CPUAndNeuralEngine; {coreml_count} CoremlOp nodes.");

    let runnable = model.into_runnable()?;
    let input = make_input()?;

    for _ in 0..5 {
        let _ = runnable.run(tvec![input.clone().into()])?;
    }

    let target_secs = 30u64;
    let start = Instant::now();
    let mut iters = 0usize;
    while start.elapsed().as_secs() < target_secs {
        let _ = runnable.run(tvec![input.clone().into()])?;
        iters += 1;
    }
    let elapsed = start.elapsed();
    println!("Ran {iters} iterations in {elapsed:?}  (mean {:?})", elapsed / iters as u32);
    Ok(())
}
