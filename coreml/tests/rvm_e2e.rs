//! Phase 4 canary: Robust Video Matting (RVM, MobileNetV3 variant) ONNX
//! through `CoremlRuntime`.
//!
//! RVM is a recurrent video-matting model from PeterL1n/RobustVideoMatting.
//! Per-frame inference signature (from the upstream ONNX export):
//!
//!   inputs:
//!     src              : [N, 3, H, W]      input frame
//!     r1i, r2i, r3i, r4i : [N, C, H', W']  recurrent hidden states (initial = zeros)
//!     downsample_ratio : [1]               internal downsampling factor
//!   outputs:
//!     fgr : [N, 3, H, W]  foreground RGB
//!     pha : [N, 1, H, W]  alpha matte
//!     r1o, r2o, r3o, r4o : updated hidden states for the next frame
//!
//! On the first frame the recurrent inputs may be `[1, 1, 1, 1]` zero
//! tensors (per PeterL1n's reference inference code) — the recurrent layers
//! handle that gracefully by initialising fresh state.
//!
//! Op histogram (raw ONNX, 353 nodes):
//!   85× Conv, 48× Constant, 47× Mul, 28× HardSigmoid, 27× Relu, 23× Concat,
//!   16× Add, 12× Shape, 12× Slice, 10× Split, 9× GlobalAveragePool, 8× Sub,
//!   7× Resize, 5× Sigmoid, 4× Expand, 4× Tanh, 3× AveragePool, 2× ReduceMean,
//!   2× Clip, 1× Div
//!
//! ```sh
//! cargo test -p tract-coreml --test rvm_e2e -- --ignored --nocapture
//! ```

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use half::f16;
use objc2_core_ml::MLComputeUnits;

use tract_core::internal::*;
use tract_core::ops::cnn::Conv;
use tract_core::transform::{ModelTransform, get_transform};
use tract_onnx::prelude::*;

use tract_coreml::{CoremlOp, CoremlTransform};

// We use the `_baked` ONNX where `downsample_ratio` is replaced with a
// Constant (DOWNSAMPLE_RATIO = 0.375). The original RVM ONNX exports
// `downsample_ratio` as a runtime input flowing into Concat-Resize-scales,
// which tract can't const-fold during `into_typed` — so the Resize op fails
// to infer output shape. Pre-baking is a one-time fix done with onnx Python
// (see notes/phase-3-progress.md or the CLI command captured in
// `notes/phase-4-rvm-baking.md`).
const MODEL_PATH: &str = "~/coding/dfn3-wasm-relaxed-simd/models/rvm_mobilenetv3_fp32_baked.onnx";
const SRC_SHAPE: [usize; 4] = [1, 3, 480, 640]; // typical SD video frame

fn resolve_path(path: &str) -> PathBuf {
    if let Some(stripped) = path.strip_prefix("~/") {
        let home = std::env::var("HOME").expect("HOME unset");
        PathBuf::from(home).join(stripped)
    } else {
        PathBuf::from(path)
    }
}

fn load_rvm_f16() -> Result<TypedModel> {
    let path = resolve_path(MODEL_PATH);
    let mut inf = tract_onnx::onnx().model_for_path(&path)?;

    // Bind the symbolic dims by setting input facts. `downsample_ratio` is a
    // length-1 F32 tensor; the recurrent states use the documented
    // `[1, 1, 1, 1]` zero-init pattern for the first frame.
    inf.set_input_fact(
        0, // src
        InferenceFact::dt_shape(
            f32::datum_type(),
            tvec![
                SRC_SHAPE[0] as i64,
                SRC_SHAPE[1] as i64,
                SRC_SHAPE[2] as i64,
                SRC_SHAPE[3] as i64
            ],
        ),
    )?;
    for slot in 1..=4 {
        inf.set_input_fact(slot, InferenceFact::dt_shape(f32::datum_type(), tvec![1i64, 1, 1, 1]))?;
    }

    inf.analyse(true)?;
    let model = inf.into_typed()?;
    println!("loaded {}: {} nodes", path.display(), model.nodes.len());

    let model = model.into_decluttered()?;
    println!("after into_decluttered: {} nodes", model.nodes.len());
    let mut model = model;
    let f16x = get_transform("f32_to_f16")?
        .ok_or_else(|| anyhow::anyhow!("f32_to_f16 transform not registered"))?;
    f16x.transform(&mut model)?;
    println!("after f32_to_f16: {} nodes", model.nodes.len());
    Ok(model)
}

#[test]
#[ignore = "loads external rvm_mobilenetv3_fp32.onnx; --ignored --nocapture"]
fn rvm_coverage_probe() -> Result<()> {
    let cpu_model = load_rvm_f16()?;

    // Pre-transform op histogram.
    let mut op_hist: std::collections::BTreeMap<String, usize> = Default::default();
    for n in &cpu_model.nodes {
        *op_hist.entry(n.op.name().to_string()).or_default() += 1;
    }
    println!("\n[op-type histogram, post-declutter+f16, {} nodes]", cpu_model.nodes.len());
    for (op, c) in &op_hist {
        println!("  {c:4}× {op}");
    }

    // Per-Conv eligibility breakdown.
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

    // Apply CoremlTransform.
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
    println!("\n[post-transform op histogram (what's still on CPU)]");
    for (op, c) in &after_hist {
        println!("  {c:4}× {op}");
    }

    // Diagnose: skip-reasons for stranded Resize / EinSum / SumPool.
    use tract_core::ops::cnn::SumPool;
    use tract_core::ops::einsum::EinSum;
    use tract_coreml::ops::avgpool::{AvgPoolAnalysis, analyse_sumpool};
    use tract_coreml::ops::einsum::analyse_einsum;
    use tract_coreml::ops::matmul::{MatMulAnalysis, analyse_matmul};
    use tract_coreml::ops::resize::{ResizeAnalysis, analyse_resize};
    use tract_onnx_opl::resize::Resize;

    println!("\n[stranded EinSum reasons]");
    for n in &coreml_model.nodes {
        if let Some(es) = n.op_as::<EinSum>() {
            let einsum_skip = match analyse_einsum(&coreml_model, n)? {
                ConvAnalysis::Translatable(_) => continue, // already translated as conv
                ConvAnalysis::Skip(r) => r,
            };
            let matmul_skip = match analyse_matmul(&coreml_model, n)? {
                MatMulAnalysis::Translatable(_) => continue,
                MatMulAnalysis::Skip(r) => r,
            };
            let in_shapes: Vec<String> = n
                .inputs
                .iter()
                .map(|i| {
                    let f = coreml_model.outlet_fact(*i).unwrap();
                    format!("{:?}@{:?}", f.shape, f.datum_type)
                })
                .collect();
            println!("  EinSum({}) {} axes={}", n.id, n.name, es.axes);
            println!("    inputs: [{}]", in_shapes.join("; "));
            println!("    einsum-skip: {einsum_skip}");
            println!("    matmul-skip: {matmul_skip}");
        }
    }
    println!("\n[stranded Resize reasons]");
    for n in &coreml_model.nodes {
        if n.op_as::<Resize>().is_some() {
            match analyse_resize(&coreml_model, n)? {
                ResizeAnalysis::Translatable(_) => continue,
                ResizeAnalysis::Skip(r) => println!("  Resize({}) {}: {r}", n.id, n.name),
            }
        }
    }
    println!("\n[stranded SumPool reasons]");
    for n in &coreml_model.nodes {
        if n.op_as::<SumPool>().is_some() {
            match analyse_sumpool(&coreml_model, n)? {
                AvgPoolAnalysis::Translatable(_) => continue,
                AvgPoolAnalysis::Skip(r) => println!("  SumPool({}) {}: {r}", n.id, n.name),
            }
        }
    }

    // Try to build a runnable.
    let coreml_runnable = match coreml_model.into_runnable() {
        Ok(r) => r,
        Err(e) => {
            println!("\n[NOTE] coreml_model.into_runnable() failed: {e}");
            return Ok(());
        }
    };

    // Build inputs: src (sin pattern), 4× recurrent zero-init at [1,1,1,1],
    // downsample_ratio = DOWNSAMPLE_RATIO.
    let n: usize = SRC_SHAPE.iter().product();
    let src_data: Vec<f16> =
        (0..n).map(|i| f16::from_f32(((i as f32) * 0.0001).sin() * 0.5)).collect();
    let src = Tensor::from_shape::<f16>(&SRC_SHAPE, &src_data)?;
    let r_init = Tensor::from_shape::<f16>(&[1, 1, 1, 1], &[f16::ZERO])?;

    let t0 = Instant::now();
    let out = coreml_runnable.run(tvec![
        src.clone().into(),
        r_init.clone().into(),
        r_init.clone().into(),
        r_init.clone().into(),
        r_init.into(),
    ])?;
    println!("\n[rvm] CoreML first run: {:?}", t0.elapsed());
    for (i, t) in out.iter().enumerate() {
        println!("  out[{i}]: shape {:?}", t.shape());
    }
    Ok(())
}

/// Release perf benchmark for RVM. CoreML-only — no CPU baseline because
/// tract's CPU Resize is F32-only (same constraint as MODNet).
///
/// RTF target reference: typical video matting deployments target 30 fps
/// (33.3 ms budget). 60 fps = 16.67 ms.
#[test]
#[ignore = "loads external rvm_mobilenetv3_fp32_baked.onnx; long run; --ignored --nocapture"]
fn rvm_perf_coreml_only() -> Result<()> {
    const ITERATIONS: usize = 50;
    const WARMUP: usize = 3;

    let mut model = load_rvm_f16()?;
    CoremlTransform::default().transform(&mut model)?;
    let coreml_runnable = model.into_runnable()?;

    let n: usize = SRC_SHAPE.iter().product();
    let src_data: Vec<f16> =
        (0..n).map(|i| f16::from_f32(((i as f32) * 0.0001).sin() * 0.5)).collect();
    let src = Tensor::from_shape::<f16>(&SRC_SHAPE, &src_data)?;
    let r_init = Tensor::from_shape::<f16>(&[1, 1, 1, 1], &[f16::ZERO])?;

    for _ in 0..WARMUP {
        let _ = coreml_runnable.run(tvec![
            src.clone().into(),
            r_init.clone().into(),
            r_init.clone().into(),
            r_init.clone().into(),
            r_init.clone().into(),
        ])?;
    }

    let t0 = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = coreml_runnable.run(tvec![
            src.clone().into(),
            r_init.clone().into(),
            r_init.clone().into(),
            r_init.clone().into(),
            r_init.clone().into(),
        ])?;
    }
    let coreml_per = t0.elapsed() / ITERATIONS as u32;
    let coreml_ms = coreml_per.as_secs_f64() * 1000.0;
    let rtf_30fps = coreml_ms / 33.333;
    let rtf_60fps = coreml_ms / 16.667;

    println!("\n=== RVM perf (n={ITERATIONS}, warmup={WARMUP}, src {SRC_SHAPE:?}) ===",);
    println!("  CoreML mean: {coreml_per:?}");
    println!("  RTF (30fps): {rtf_30fps:.4}");
    println!("  RTF (60fps): {rtf_60fps:.4}");
    Ok(())
}

/// CPU-vs-CoreML perf comparison on the F16 path.
///
/// Unblocked by the local Resize-F16 patch in `tract-onnx-opl/src/resize.rs`.
/// RVM contains an internal Resize so this previously errored with
/// "Tensor datum type error: tensor is F16, accessed as F32" inside Resize::eval.
#[test]
#[ignore = "loads external rvm_mobilenetv3_fp32_baked.onnx; long run; --ignored --nocapture"]
fn rvm_perf_cpu_vs_coreml() -> Result<()> {
    const ITERATIONS_COREML: usize = 50;
    const WARMUP_COREML: usize = 3;
    const ITERATIONS_CPU: usize = 3;
    const WARMUP_CPU: usize = 1;

    let cpu_model = load_rvm_f16()?;
    let mut coreml_model = load_rvm_f16()?;
    CoremlTransform::default().transform(&mut coreml_model)?;
    let cpu_runnable = cpu_model.into_runnable()?;
    let coreml_runnable = coreml_model.into_runnable()?;

    let n: usize = SRC_SHAPE.iter().product();
    let src_data: Vec<f16> =
        (0..n).map(|i| f16::from_f32(((i as f32) * 0.0001).sin() * 0.5)).collect();
    let src = Tensor::from_shape::<f16>(&SRC_SHAPE, &src_data)?;
    let r_init = Tensor::from_shape::<f16>(&[1, 1, 1, 1], &[f16::ZERO])?;
    let inputs = || -> TVec<TValue> {
        tvec![
            src.clone().into(),
            r_init.clone().into(),
            r_init.clone().into(),
            r_init.clone().into(),
            r_init.clone().into(),
        ]
    };

    println!("\n[rvm] warming CPU path (F16; this is the path Resize-F16 patch unblocks)");
    for _ in 0..WARMUP_CPU {
        let _ = cpu_runnable.run(inputs())?;
    }
    let t0 = Instant::now();
    for _ in 0..ITERATIONS_CPU {
        let _ = cpu_runnable.run(inputs())?;
    }
    let cpu_per = t0.elapsed() / ITERATIONS_CPU as u32;
    let cpu_ms = cpu_per.as_secs_f64() * 1000.0;

    println!("[rvm] warming CoreML path");
    for _ in 0..WARMUP_COREML {
        let _ = coreml_runnable.run(inputs())?;
    }
    let t0 = Instant::now();
    for _ in 0..ITERATIONS_COREML {
        let _ = coreml_runnable.run(inputs())?;
    }
    let coreml_per = t0.elapsed() / ITERATIONS_COREML as u32;
    let coreml_ms = coreml_per.as_secs_f64() * 1000.0;

    let speedup = cpu_ms / coreml_ms;
    let rtf_60fps = coreml_ms / 16.667;
    println!("\n=== RVM CPU-vs-CoreML perf (src {SRC_SHAPE:?}, F16) ===");
    println!("  Tract CPU mean (n={ITERATIONS_CPU}): {cpu_per:?}  = {cpu_ms:.2} ms");
    println!("  CoreML mean    (n={ITERATIONS_COREML}): {coreml_per:?}  = {coreml_ms:.2} ms");
    println!("  Speedup:        {speedup:.1}×");
    println!("  RTF (60fps):    {rtf_60fps:.4}");

    Ok(())
}

// One test per compute-unit mode. SIGABRT from CoreML/MPS bypasses Rust
// panic handling, so we can't combine into a single test — the abort would
// kill all subsequent measurements. With separate tests, a failing mode
// just shows up as one failed test and the others still report numbers.
//
// Question we're answering: "Does RVM engage ANE?" Previously documented
// hypothesis: "recurrent zero-init [1,1,1,1] state shapes confuse Apple's
// scheduler". With per-mode results we can compare CPUAndNeuralEngine vs
// CPUAndGPU vs All directly. If ANE engages, CPUAndNeuralEngine ≪ CPUOnly.

const RVM_PERF_ITERATIONS: usize = 30;
const RVM_PERF_WARMUP: usize = 3;

fn rvm_bench_one(units: MLComputeUnits, label: &str) -> Result<()> {
    let n: usize = SRC_SHAPE.iter().product();
    let src_data: Vec<f16> =
        (0..n).map(|i| f16::from_f32(((i as f32) * 0.0001).sin() * 0.5)).collect();
    let src = Tensor::from_shape::<f16>(&SRC_SHAPE, &src_data)?;
    let r_init = Tensor::from_shape::<f16>(&[1, 1, 1, 1], &[f16::ZERO])?;

    let mut model = load_rvm_f16()?;
    let xform = CoremlTransform { compute_units: units };
    xform.transform(&mut model)?;
    let count = model.nodes.iter().filter(|n| n.op_as::<CoremlOp>().is_some()).count();
    println!("[{label}] {count} CoremlOp nodes after transform");
    let runnable = model.into_runnable()?;
    for _ in 0..RVM_PERF_WARMUP {
        let _ = runnable.run(tvec![
            src.clone().into(),
            r_init.clone().into(),
            r_init.clone().into(),
            r_init.clone().into(),
            r_init.clone().into(),
        ])?;
    }
    let t0 = Instant::now();
    for _ in 0..RVM_PERF_ITERATIONS {
        let _ = runnable.run(tvec![
            src.clone().into(),
            r_init.clone().into(),
            r_init.clone().into(),
            r_init.clone().into(),
            r_init.clone().into(),
        ])?;
    }
    let per = t0.elapsed() / RVM_PERF_ITERATIONS as u32;
    let ms = per.as_secs_f64() * 1000.0;
    println!("\n=== RVM perf [{label}] (n={RVM_PERF_ITERATIONS}, warmup={RVM_PERF_WARMUP}) ===");
    println!("  mean: {ms:.3} ms");
    Ok(())
}

#[test]
#[ignore = "loads external rvm; release perf only; --ignored --nocapture"]
fn rvm_perf_cpu_only() -> Result<()> {
    rvm_bench_one(MLComputeUnits::CPUOnly, "CPUOnly")
}
#[test]
#[ignore = "loads external rvm; release perf only; --ignored --nocapture"]
fn rvm_perf_cpu_and_gpu() -> Result<()> {
    rvm_bench_one(MLComputeUnits::CPUAndGPU, "CPUAndGPU")
}
#[test]
#[ignore = "loads external rvm; release perf only; --ignored --nocapture"]
fn rvm_perf_cpu_and_ane() -> Result<()> {
    rvm_bench_one(MLComputeUnits::CPUAndNeuralEngine, "CPUAndNeuralEngine")
}
#[test]
#[ignore = "loads external rvm; release perf only; --ignored --nocapture"]
fn rvm_perf_all() -> Result<()> {
    rvm_bench_one(MLComputeUnits::All, "All")
}

/// Sustained loop for `powermetrics` ANE measurement (pair with
/// `sudo powermetrics -A --show-extra-power-info` in a second shell).
#[test]
#[ignore = "loads external rvm; long sustained run; pair with sudo powermetrics"]
fn rvm_sustained_for_powermetrics() -> Result<()> {
    let mut model = load_rvm_f16()?;
    let xform = CoremlTransform { compute_units: MLComputeUnits::CPUAndNeuralEngine };
    xform.transform(&mut model)?;

    let coreml_count = model.nodes.iter().filter(|n| n.op_as::<CoremlOp>().is_some()).count();
    println!("Compiled with CPUAndNeuralEngine; {coreml_count} CoremlOp nodes.");

    let runnable = model.into_runnable()?;
    let n: usize = SRC_SHAPE.iter().product();
    let src_data: Vec<f16> =
        (0..n).map(|i| f16::from_f32(((i as f32) * 0.0001).sin() * 0.5)).collect();
    let src = Tensor::from_shape::<f16>(&SRC_SHAPE, &src_data)?;
    let r_init = Tensor::from_shape::<f16>(&[1, 1, 1, 1], &[f16::ZERO])?;

    for _ in 0..3 {
        let _ = runnable.run(tvec![
            src.clone().into(),
            r_init.clone().into(),
            r_init.clone().into(),
            r_init.clone().into(),
            r_init.clone().into(),
        ])?;
    }

    let target_secs = 30u64;
    let start = Instant::now();
    let mut iters = 0usize;
    while start.elapsed().as_secs() < target_secs {
        let _ = runnable.run(tvec![
            src.clone().into(),
            r_init.clone().into(),
            r_init.clone().into(),
            r_init.clone().into(),
            r_init.clone().into(),
        ])?;
        iters += 1;
    }
    let elapsed = start.elapsed();
    println!("Ran {iters} iterations in {elapsed:?}  (mean {:?})", elapsed / iters as u32);
    Ok(())
}
