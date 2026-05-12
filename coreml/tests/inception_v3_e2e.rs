//! Phase 3 canary #8: InceptionV3 ONNX through `CoremlRuntime`.
//!
//! Pure CNN classifier, similar shape to MobileNet (input `[1,3,299,299]` →
//! output `[1,1000]`) but ~24M params vs MobileNet's ~3M, and a much heavier
//! op mix (more Conv branches, AvgPool/MaxPool, Concat across the Inception
//! modules). Source ONNX is the timm Opset16 export from the ONNX model zoo.
//!
//! `#[ignore]` by default; loads model from outside the workspace. Run with:
//!
//! ```sh
//! cargo test -p tract-coreml --test inception_v3_e2e -- --ignored --nocapture
//! ```
//!
//! Three tests:
//! - `inception_v3_coverage_probe`: per-Conv translatability + op histograms
//!   (pre/post `CoremlTransform`) + stranded-op reasons + first-run latency.
//! - `inception_v3_perf_coreml_only`: release-mode CoreML perf (n=100 warmup=3)
//!   reporting per-iter latency + RTF @ 30 fps.
//! - `inception_v3_sustained_for_powermetrics`: 30 s loop pinned to
//!   `MLComputeUnits::CPUAndNeuralEngine` for paired `sudo powermetrics`
//!   captures.

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

const MODEL_PATH: &str = "~/coding/dfn3-wasm-relaxed-simd/models/inception_v3.onnx";
const INPUT_SHAPE: [usize; 4] = [1, 3, 299, 299];

fn resolve_path(path: &str) -> PathBuf {
    if let Some(stripped) = path.strip_prefix("~/") {
        let home = std::env::var("HOME").expect("HOME unset");
        PathBuf::from(home).join(stripped)
    } else {
        PathBuf::from(path)
    }
}

/// Load inception_v3.onnx, declutter (folds BN+ReLU into Conv but keeps Conv
/// as Conv), then cast to F16. Mirrors the load shape used by the MobileNet
/// canary: do *not* `into_optimized()` here — that lowers `Conv` to
/// `Im2Col + MatMul` and the translator never sees the high-level Conv op.
fn load_inception_v3_f16() -> Result<TypedModel> {
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

/// Build a deterministic F16 input tensor of `INPUT_SHAPE`.
fn make_input() -> Result<Tensor> {
    let n: usize = INPUT_SHAPE.iter().product();
    let data: Vec<f16> = (0..n).map(|i| f16::from_f32(((i as f32) * 0.001).sin() * 0.5)).collect();
    Tensor::from_shape::<f16>(&INPUT_SHAPE, &data)
}

#[test]
#[ignore = "loads external inception_v3.onnx; run with --ignored --nocapture"]
fn inception_v3_coverage_probe() -> Result<()> {
    let cpu_model = load_inception_v3_f16()?;

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

    // Op-type histogram of remaining (= what's still on CPU = stranded ops).
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

    // End-to-end first run on the CoreML-dispatched path. If any leftover op
    // isn't supported the runnable will fail to build / run; either outcome is
    // useful coverage signal.
    let input = make_input()?;
    let coreml_runnable = match coreml_model.into_runnable() {
        Ok(r) => r,
        Err(e) => {
            println!("\n[NOTE] coreml_model.into_runnable() failed: {e}");
            println!(
                "[NOTE] this typically means a CPU-fallback op needs work; \
                     skipping end-to-end run"
            );
            return Ok(());
        }
    };
    let t0 = Instant::now();
    let coreml_out = coreml_runnable.run(tvec![input.clone().into()])?;
    println!("CoreML first run: {:?}", t0.elapsed());

    assert_eq!(coreml_out.len(), 1);
    let k = &coreml_out[0];
    println!("CoreML output shape: {:?}", k.shape());
    assert_eq!(k.shape(), &[1, 1000]);

    Ok(())
}

#[test]
#[ignore = "loads external inception_v3.onnx; release perf only; --ignored --nocapture"]
fn inception_v3_perf_coreml_only() -> Result<()> {
    const ITERATIONS: usize = 100;
    const WARMUP: usize = 3;

    let mut coreml_model = load_inception_v3_f16()?;
    CoremlTransform::default().transform(&mut coreml_model)?;

    let coreml_count =
        coreml_model.nodes.iter().filter(|n| n.op_as::<CoremlOp>().is_some()).count();
    println!("CoremlOp nodes after transform: {coreml_count}");

    let coreml_runnable = coreml_model.into_runnable()?;
    let input = make_input()?;

    for _ in 0..WARMUP {
        let _ = coreml_runnable.run(tvec![input.clone().into()])?;
    }

    let t0 = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = coreml_runnable.run(tvec![input.clone().into()])?;
    }
    let coreml_per = t0.elapsed() / ITERATIONS as u32;
    let coreml_ms = coreml_per.as_secs_f64() * 1000.0;
    let rtf_30fps = coreml_ms / 33.333;

    println!("\n=== InceptionV3 perf (CoreML only, n={ITERATIONS}, warmup={WARMUP}) ===");
    println!("  CoreML mean: {coreml_per:?}  ({coreml_ms:.3} ms)");
    println!("  RTF (30fps): {rtf_30fps:.4}");

    Ok(())
}

/// CPU-vs-CoreML perf comparison on the F16 path.
///
/// Unblocked by the local Resize-F16 patch in `tract-onnx-opl/src/resize.rs`.
/// InceptionV3 contains a Resize in the head so this previously errored with
/// "Tensor datum type error: tensor is F16, accessed as F32" inside Resize::eval.
#[test]
#[ignore = "loads external inception_v3.onnx; release perf only; --ignored --nocapture"]
fn inception_v3_perf_cpu_vs_coreml() -> Result<()> {
    const ITERATIONS_COREML: usize = 100;
    const WARMUP_COREML: usize = 3;
    const ITERATIONS_CPU: usize = 10;
    const WARMUP_CPU: usize = 1;

    let cpu_model = load_inception_v3_f16()?;
    let mut coreml_model = load_inception_v3_f16()?;
    CoremlTransform::default().transform(&mut coreml_model)?;
    let cpu_runnable = cpu_model.into_runnable()?;
    let coreml_runnable = coreml_model.into_runnable()?;
    let input = make_input()?;

    println!("\n[inception_v3] warming CPU path (F16; this is the path Resize-F16 patch unblocks)");
    for _ in 0..WARMUP_CPU {
        let _ = cpu_runnable.run(tvec![input.clone().into()])?;
    }
    let t0 = Instant::now();
    for _ in 0..ITERATIONS_CPU {
        let _ = cpu_runnable.run(tvec![input.clone().into()])?;
    }
    let cpu_per = t0.elapsed() / ITERATIONS_CPU as u32;
    let cpu_ms = cpu_per.as_secs_f64() * 1000.0;

    println!("[inception_v3] warming CoreML path");
    for _ in 0..WARMUP_COREML {
        let _ = coreml_runnable.run(tvec![input.clone().into()])?;
    }
    let t0 = Instant::now();
    for _ in 0..ITERATIONS_COREML {
        let _ = coreml_runnable.run(tvec![input.clone().into()])?;
    }
    let coreml_per = t0.elapsed() / ITERATIONS_COREML as u32;
    let coreml_ms = coreml_per.as_secs_f64() * 1000.0;

    let speedup = cpu_ms / coreml_ms;
    let rtf_30fps = coreml_ms / 33.333;
    println!("\n=== InceptionV3 CPU-vs-CoreML perf (F16) ===");
    println!("  Tract CPU mean (n={ITERATIONS_CPU}): {cpu_per:?}  = {cpu_ms:.2} ms");
    println!("  CoreML mean    (n={ITERATIONS_COREML}): {coreml_per:?}  = {coreml_ms:.2} ms");
    println!("  Speedup:        {speedup:.1}×");
    println!("  RTF (30fps):    {rtf_30fps:.4}");

    Ok(())
}

/// A/B compute-unit comparison — runs the same compiled-once model under
/// CPUOnly, CPUAndGPU, CPUAndNeuralEngine, and All and reports per-iter
/// latency for each. Lets us infer ANE engagement without `sudo
/// powermetrics`: if All ≈ CPUAndNeuralEngine and both << CPUOnly, the
/// ANE is being scheduled. If CPUAndGPU ≈ All, the GPU path is competitive
/// and ANE may not be exclusive.
#[test]
#[ignore = "loads external inception_v3.onnx; release perf only; --ignored --nocapture"]
fn inception_v3_perf_compute_units_compare() -> Result<()> {
    const ITERATIONS: usize = 50;
    const WARMUP: usize = 3;

    let input = make_input()?;
    let bench_units = |units: MLComputeUnits, label: &str| -> Result<f64> {
        let mut model = load_inception_v3_f16()?;
        let xform = CoremlTransform { compute_units: units };
        xform.transform(&mut model)?;
        let count = model.nodes.iter().filter(|n| n.op_as::<CoremlOp>().is_some()).count();
        println!("[{label}] {count} CoremlOp nodes after transform");
        let runnable = model.into_runnable()?;
        for _ in 0..WARMUP {
            let _ = runnable.run(tvec![input.clone().into()])?;
        }
        let t0 = Instant::now();
        for _ in 0..ITERATIONS {
            let _ = runnable.run(tvec![input.clone().into()])?;
        }
        let per = t0.elapsed() / ITERATIONS as u32;
        let ms = per.as_secs_f64() * 1000.0;
        println!("[{label}] mean: {ms:.3} ms");
        Ok(ms)
    };

    let cpu_only = bench_units(MLComputeUnits::CPUOnly, "CPUOnly")?;
    let cpu_gpu = bench_units(MLComputeUnits::CPUAndGPU, "CPUAndGPU")?;
    let cpu_ane = bench_units(MLComputeUnits::CPUAndNeuralEngine, "CPUAndNeuralEngine")?;
    let all = bench_units(MLComputeUnits::All, "All")?;

    println!("\n=== InceptionV3 compute-unit comparison (n={ITERATIONS}, warmup={WARMUP}) ===");
    println!("  CPUOnly             : {cpu_only:.3} ms  (1.00× baseline)");
    println!("  CPUAndGPU           : {cpu_gpu:.3} ms  ({:.2}× vs CPUOnly)", cpu_only / cpu_gpu);
    println!("  CPUAndNeuralEngine  : {cpu_ane:.3} ms  ({:.2}× vs CPUOnly)", cpu_only / cpu_ane);
    println!("  All                 : {all:.3} ms  ({:.2}× vs CPUOnly)", cpu_only / all);

    Ok(())
}

/// Sustained-loop benchmark for `powermetrics` ANE measurement. Runs the
/// CoreML-dispatched InceptionV3 for ~30 s pinned to `CPUAndNeuralEngine` so
/// a paired `sudo powermetrics -A --show-extra-power-info` capture in another
/// terminal can record ANE Power.
#[test]
#[ignore = "loads external inception_v3.onnx; long sustained run; pair with sudo powermetrics"]
fn inception_v3_sustained_for_powermetrics() -> Result<()> {
    let mut model = load_inception_v3_f16()?;
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
