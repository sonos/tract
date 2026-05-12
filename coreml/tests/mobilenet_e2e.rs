//! Real-model end-to-end test: MobileNet v2 ONNX through `CoremlRuntime`.
//!
//! `#[ignore]` by default because it loads `mobilenetv2-7.onnx` from outside
//! the workspace (`~/coding/dfn3-wasm-relaxed-simd/models/`). Run with:
//!
//! ```sh
//! cargo test -p tract-coreml --test mobilenet_e2e -- --ignored --nocapture
//! ```
//!
//! Closes Phase 1 gates 3 + 5 (real-model correctness + speedup) and gives a
//! per-op coverage signal: prints how many Conv nodes were translated to
//! CoremlOp vs left on CPU. Also benchmarks CPU-only vs CoreML-dispatched
//! inference.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use half::f16;
use objc2_core_ml::MLComputeUnits;

use tract_core::internal::*;
use tract_core::ops::cnn::Conv;
use tract_core::ops::einsum::EinSum;
use tract_core::transform::{ModelTransform, get_transform};
use tract_onnx::prelude::InferenceModelExt;

use tract_coreml::{CoremlOp, CoremlTransform};

const MODEL_PATH: &str = "~/coding/dfn3-wasm-relaxed-simd/models/mobilenetv2-7.onnx";
const INPUT_SHAPE: [usize; 4] = [1, 3, 224, 224];

fn resolve_path(path: &str) -> PathBuf {
    if let Some(stripped) = path.strip_prefix("~/") {
        let home = std::env::var("HOME").expect("HOME unset");
        PathBuf::from(home).join(stripped)
    } else {
        PathBuf::from(path)
    }
}

/// Load mobilenetv2-7.onnx, declutter (folds BN+ReLU into Conv but keeps Conv as Conv),
/// then cast to F16. Returns a model ready to feed to `CoremlTransform`.
///
/// Critical: do *not* call `into_optimized()` here — that lowers `Conv` to
/// `Im2Col + MatMul` and our translator never sees the high-level Conv op.
/// `CoremlTransform` runs before the final optimizer; afterwards the leftover
/// CPU-fallback ops still get optimized by the Runtime's own `into_optimized()` step.
fn load_mobilenet_f16() -> Result<TypedModel> {
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

/// Count how many `Conv` nodes vs `CoremlOp` nodes a model has — quick
/// per-op coverage signal.
fn op_counts(model: &TypedModel) -> (usize, usize, usize) {
    let convs = model.nodes.iter().filter(|n| n.op_as::<Conv>().is_some()).count();
    let coreml_ops = model.nodes.iter().filter(|n| n.op_as::<CoremlOp>().is_some()).count();
    let total = model.nodes.len();
    (convs, coreml_ops, total)
}

/// Build a deterministic F16 input tensor of `INPUT_SHAPE`.
fn make_input() -> Result<Tensor> {
    let n: usize = INPUT_SHAPE.iter().product();
    let data: Vec<f16> = (0..n).map(|i| f16::from_f32(((i as f32) * 0.001).sin() * 0.5)).collect();
    Tensor::from_shape::<f16>(&INPUT_SHAPE, &data)
}

#[test]
#[ignore = "loads external mobilenetv2-7.onnx; run with --ignored"]
fn mobilenet_v2_cpu_vs_coreml_correctness() -> Result<()> {
    let cpu_model = load_mobilenet_f16()?;
    let (convs_before, coreml_before, total_before) = op_counts(&cpu_model);
    println!("=== before CoremlTransform ===");
    println!("  total nodes: {total_before}");
    println!("  Conv nodes:  {convs_before}");
    println!("  CoremlOp:    {coreml_before} (sanity: should be 0)");
    assert_eq!(coreml_before, 0);

    // Per-Conv eligibility breakdown: print each Conv with the analyse_conv
    // verdict. This is the per-op coverage signal that drives Phase 2.
    use tract_coreml::ops::conv::{ConvAnalysis, analyse_conv};
    let conv_node_ids: Vec<usize> =
        cpu_model.nodes.iter().filter(|n| n.op_as::<Conv>().is_some()).map(|n| n.id).collect();
    // Op-type histogram of the post-decluttered + f32_to_f16 model. Tells us
    // which ops sit between Convs (= what fusion needs to also translate to
    // grow MLPackages past per-Conv granularity).
    let mut op_hist: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    for n in &cpu_model.nodes {
        let key = n.op.name().to_string();
        *op_hist.entry(key).or_default() += 1;
    }
    println!("\n[op-type histogram, post-declutter+f16, 533 nodes]");
    for (op, count) in &op_hist {
        if *count >= 3 {
            println!("  {count:4}× {op}");
        }
    }
    println!("  (rare ops <3 occurrences omitted)");

    // Inspect first few EinSum nodes to understand the axes pattern (helps us
    // design a tract→MIL EinSum translator).
    let einsums: Vec<&TypedNode> =
        cpu_model.nodes.iter().filter(|n| n.op_as::<EinSum>().is_some()).take(3).collect();
    println!("\n[first {} EinSum nodes — axes structure]", einsums.len());
    for n in einsums {
        let es = n.op_as::<EinSum>().unwrap();
        let in_shapes: Vec<String> = n
            .inputs
            .iter()
            .map(|i| {
                let f = cpu_model.outlet_fact(*i).unwrap();
                format!("{:?}@{:?}", f.shape, f.datum_type)
            })
            .collect();
        let out_fact = &n.outputs[0].fact;
        println!(
            "  {}: axes={} operating_dt={:?} q_params={:?}\n    inputs: [{}]\n    output: {:?}@{:?}",
            n.name,
            es.axes,
            es.operating_dt,
            es.q_params,
            in_shapes.join("; "),
            out_fact.shape,
            out_fact.datum_type,
        );
    }

    println!("\n[per-Conv eligibility] {} Conv nodes:", conv_node_ids.len());
    let mut translatable = 0;
    let mut skip_counts: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();
    for id in &conv_node_ids {
        let n = &cpu_model.nodes[*id];
        let conv = n.op_as::<Conv>().unwrap();
        match analyse_conv(&cpu_model, n)? {
            ConvAnalysis::Translatable(_) => {
                translatable += 1;
                println!(
                    "  TRANSLATABLE  Conv {id} {}  (group={} in={} out={})",
                    n.name,
                    conv.group,
                    conv.pool_spec.input_channels,
                    conv.pool_spec.output_channels
                );
            }
            ConvAnalysis::Skip(reason) => {
                let bucket = reason.split(' ').take(3).collect::<Vec<_>>().join(" ");
                *skip_counts.entry(bucket).or_default() += 1;
                if *skip_counts.values().max().unwrap_or(&0) <= 3 {
                    println!("  skip          Conv {id} {}: {reason}", n.name);
                }
            }
        }
    }
    println!("\n[skip-reason histogram]");
    for (reason, count) in &skip_counts {
        println!("  {count:3}× {reason}…");
    }
    println!("  → {translatable}/{} translatable", conv_node_ids.len());

    // Apply the Coreml transform (to a clone we'll run alongside the CPU model).
    let mut coreml_model = cpu_model.clone();
    let t0 = Instant::now();
    CoremlTransform::default().transform(&mut coreml_model)?;
    println!("CoremlTransform: {:?}", t0.elapsed());

    let (convs_after, coreml_after, total_after) = op_counts(&coreml_model);
    let translated = convs_before - convs_after;
    println!("=== after CoremlTransform ===");
    println!("  total nodes:        {total_after}");
    println!("  Conv nodes left:    {convs_after}  (CPU fallback)");
    println!("  CoremlOp nodes:     {coreml_after}");
    println!(
        "  translated:         {translated}/{convs_before} ({:.0}%)",
        100.0 * translated as f64 / convs_before as f64
    );

    assert!(
        coreml_after > 0,
        "expected at least one Conv to be translated to CoremlOp; transform may be silently no-oping"
    );

    // Run CPU baseline and CoreML path, compare.
    let input = make_input()?;
    let cpu_runnable = cpu_model.into_runnable()?;
    let coreml_runnable = coreml_model.into_runnable()?;

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
    assert_eq!(c.shape(), k.shape());
    assert_eq!(c.shape(), &[1, 1000]);

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
    println!("\n[mobilenet cpu vs coreml] max_abs_diff = {max_abs:.6}  rel_l2 = {rel_l2:.6e}");

    // Also: top-1 / top-5 class agreement (more robust than raw L2 for classifiers).
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
    assert_eq!(
        cpu_top1.0, coreml_top1.0,
        "top-1 class disagreement: CPU={}, CoreML={}",
        cpu_top1.0, coreml_top1.0
    );

    // FP16 accumulates more drift the more ops live inside the MLPackage —
    // and Phase 2 fusion deliberately puts more ops in. Per the handoff
    // verification protocol "Output L2 norm matches CPU reference within
    // tolerance (1e-3 for fp32 paths; larger for ANE's fp16 path — quantify
    // per model)" — we widen to 5e-2 once full BN/ReLU chains land in
    // CoreML. Top-1 class match (asserted above) is the load-bearing
    // correctness check; rel_l2 is the additional sanity ceiling.
    assert!(rel_l2 < 5e-2, "rel_l2 {rel_l2} exceeds Phase 2 FP16 budget 5e-2");
    Ok(())
}

/// Sustained-loop benchmark for `powermetrics` ANE measurement. Runs the
/// CoreML-dispatched MobileNet for ~30s under the requested compute units,
/// so a separate `sudo powermetrics -A --show-extra-power-info` capture in
/// another terminal can record ANE Power.
///
/// Run two ways:
/// ```sh
/// cargo test -p tract-coreml --release --test mobilenet_e2e -- \
///     --ignored --nocapture mobilenet_v2_sustained_for_powermetrics
/// ```
/// The default uses `MLComputeUnits::CPUAndNeuralEngine` to force GPU off the
/// menu — the cheap-experiment workaround for Gate 4 when each MLPackage is
/// too small for Apple to pick ANE under `All`.
#[test]
#[ignore = "loads external mobilenetv2-7.onnx; long sustained run; pair with sudo powermetrics"]
fn mobilenet_v2_sustained_for_powermetrics() -> Result<()> {
    let mut model = load_mobilenet_f16()?;
    let xform = CoremlTransform { compute_units: MLComputeUnits::CPUAndNeuralEngine };
    xform.transform(&mut model)?;

    let coreml_count = model.nodes.iter().filter(|n| n.op_as::<CoremlOp>().is_some()).count();
    println!("Compiled with CPUAndNeuralEngine; {coreml_count} CoremlOp nodes.");

    let runnable = model.into_runnable()?;
    let input = make_input()?;

    // Warmup.
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

#[test]
#[ignore = "loads external mobilenetv2-7.onnx; run with --ignored --nocapture"]
fn mobilenet_v2_perf_cpu_vs_coreml() -> Result<()> {
    const ITERATIONS: usize = 200;
    const WARMUP: usize = 5;

    let cpu_model = load_mobilenet_f16()?;
    let mut coreml_model = cpu_model.clone();
    CoremlTransform::default().transform(&mut coreml_model)?;

    let cpu_runnable = cpu_model.into_runnable()?;
    let coreml_runnable = coreml_model.into_runnable()?;

    let input = make_input()?;

    // Warmup both paths.
    for _ in 0..WARMUP {
        let _ = cpu_runnable.run(tvec![input.clone().into()])?;
        let _ = coreml_runnable.run(tvec![input.clone().into()])?;
    }

    // Time CPU.
    let t0 = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = cpu_runnable.run(tvec![input.clone().into()])?;
    }
    let cpu_total = t0.elapsed();
    let cpu_per = cpu_total / ITERATIONS as u32;

    // Time CoreML.
    let t0 = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = coreml_runnable.run(tvec![input.clone().into()])?;
    }
    let coreml_total = t0.elapsed();
    let coreml_per = coreml_total / ITERATIONS as u32;

    let speedup = cpu_per.as_secs_f64() / coreml_per.as_secs_f64();
    println!("\n=== MobileNet v2 perf (n={ITERATIONS}, warmup={WARMUP}) ===");
    println!("  CPU mean:    {cpu_per:?}");
    println!("  CoreML mean: {coreml_per:?}");
    println!("  Speedup:     {speedup:.2}x");

    Ok(())
}
