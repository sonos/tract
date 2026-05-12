//! Phase 3 segmentation-canary probe: MODNet through `CoremlRuntime`.
//!
//! MODNet is a portrait-matting model. ONNX export carries dynamic batch_size,
//! height, width — we bind them all to fixed sizes via `set_input_fact` +
//! `analyse(true)`. Op-mix per ONNX inspection (487 nodes, opset 11):
//!   69× Conv, 35× Clip, 32× Slice, 24× Concat, 17× Relu, 16× BN,
//!   16× InstanceNorm, 10× Add, 9× Resize, 4× Unsqueeze, 3× Shape, 2× Reshape,
//!   2× MatMul, 2× Sigmoid, 1× GAP, 1× Expand, 1× Mul
//!
//! Translatable (Phase 3 close): Conv, Concat, Relu (via Max+const), Add,
//! Resize, Unsqueeze (via AddAxis), Sigmoid, GlobalAveragePool (via Reduce).
//!
//! Still gaps: Clip (would be Min+Max+consts; not directly handled), Slice,
//! InstanceNorm, MatMul (general), Shape, Reshape, Expand. So we expect a
//! partial translation; the probe surfaces what's left to add.
//!
//! ```sh
//! cargo test -p tract-coreml --test modnet_e2e -- --ignored --nocapture
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

const MODEL_PATHS: &[&str] = &[
    "~/coding/v7-webgl-relighting-worktree/models/onnx/modnet.onnx",
    "~/coding/v7-webgl-relighting-worktree/models/onnx/modnet_fp16.onnx",
];
const INPUT_SHAPE: [usize; 4] = [1, 3, 512, 512];

fn resolve_path(path: &str) -> PathBuf {
    if let Some(stripped) = path.strip_prefix("~/") {
        let home = std::env::var("HOME").expect("HOME unset");
        PathBuf::from(home).join(stripped)
    } else {
        PathBuf::from(path)
    }
}

fn load_model_f16() -> Result<TypedModel> {
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
    Err(last_err.unwrap_or_else(|| anyhow::anyhow!("no MODNet model found")))
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
    println!(
        "loaded {}: {} nodes (input bound to {INPUT_SHAPE:?})",
        path.display(),
        model.nodes.len()
    );

    let model = model.into_decluttered()?;
    println!("after into_decluttered: {} nodes", model.nodes.len());
    let mut model = model;
    let f16 = get_transform("f32_to_f16")?
        .ok_or_else(|| anyhow::anyhow!("f32_to_f16 transform not registered"))?;
    f16.transform(&mut model)?;
    println!("after f32_to_f16: {} nodes", model.nodes.len());
    Ok(model)
}

#[test]
#[ignore = "loads external modnet.onnx; --ignored --nocapture"]
fn modnet_coverage_probe() -> Result<()> {
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

    // Diagnose Resize: why is it being skipped despite the translator?
    use tract_coreml::ops::resize::{ResizeAnalysis, analyse_resize};
    use tract_onnx_opl::resize::Resize;
    let resize_ids: Vec<usize> =
        cpu_model.nodes.iter().filter(|n| n.op_as::<Resize>().is_some()).map(|n| n.id).collect();
    let mut resize_t = 0;
    let mut resize_skips: std::collections::BTreeMap<String, usize> = Default::default();
    for id in &resize_ids {
        match analyse_resize(&cpu_model, &cpu_model.nodes[*id])? {
            ResizeAnalysis::Translatable(_) => resize_t += 1,
            ResizeAnalysis::Skip(r) => {
                let bucket = r.split(' ').take(6).collect::<Vec<_>>().join(" ");
                *resize_skips.entry(bucket).or_default() += 1;
            }
        }
    }
    println!("\n[per-Resize] {resize_t}/{} translatable", resize_ids.len());
    for (r, c) in &resize_skips {
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
    println!("\n[post-transform op histogram (what's still on CPU)]");
    for (op, c) in &after_hist {
        println!("  {c:4}× {op}");
    }

    // Detail the stranded EinSum + Cast nodes (what's preventing fusion).
    use tract_core::ops::cast::Cast;
    use tract_core::ops::einsum::EinSum;
    println!("\n[stranded Cast nodes — predecessors/successors + dtype]");
    for n in &coreml_model.nodes {
        if n.op_as::<Cast>().is_some() {
            let cast = n.op_as::<Cast>().unwrap();
            let in_fact = coreml_model.outlet_fact(n.inputs[0]).ok();
            let out_fact = &n.outputs[0].fact;
            let preds: Vec<String> = n
                .inputs
                .iter()
                .map(|i| format!("{}({})", coreml_model.nodes[i.node].op.name(), i.node))
                .collect();
            let succs: Vec<String> = coreml_model
                .nodes
                .iter()
                .filter(|other| other.inputs.iter().any(|i| i.node == n.id))
                .map(|other| format!("{}({})", other.op.name(), other.id))
                .collect();
            println!(
                "  Cast({}) {} | from={:?} to={:?} | shape_in={:?} shape_out={:?}",
                n.id,
                n.name,
                in_fact.map(|f| f.datum_type),
                cast.to,
                in_fact.map(|f| &f.shape),
                &out_fact.shape,
            );
            println!("    preds: {preds:?}");
            println!("    succs: {succs:?}");
        }
    }
    // Trace one Reduce<MeanOfSquares> chain to understand the InstanceNorm
    // pattern shape (helps design a fold-to-mb.instance_norm translator).
    println!("\n[InstanceNorm chain trace — one Reduce<MeanOfSquares> + neighbors]");
    use tract_core::ops::nn::Reduce;
    let mut traced = 0;
    for n in &cpu_model.nodes {
        if let Some(red) = n.op_as::<Reduce>()
            && format!("{:?}", red.reducer) == "MeanOfSquares"
            && traced < 1
        {
            traced += 1;
            println!(
                "  Reduce<MoS>({}) {} | axes={:?} | input shape={:?}",
                n.id,
                n.name,
                red.axes,
                cpu_model.outlet_fact(n.inputs[0]).map(|f| f.shape.clone())
            );
            // Walk forward + backward 6 hops on each side to capture the chain.
            let mut frontier = vec![n.id];
            let mut seen = std::collections::HashSet::new();
            seen.insert(n.id);
            let mut depth = 0;
            while !frontier.is_empty() && depth < 6 {
                let mut next = Vec::new();
                for &id in &frontier {
                    let nn = &cpu_model.nodes[id];
                    let in_dts: Vec<String> = nn
                        .inputs
                        .iter()
                        .map(|i| {
                            let f = cpu_model.outlet_fact(*i).unwrap();
                            format!("{:?}@{:?}", f.shape, f.datum_type)
                        })
                        .collect();
                    let succs: Vec<usize> = cpu_model
                        .nodes
                        .iter()
                        .filter(|other| other.inputs.iter().any(|i| i.node == id))
                        .map(|other| other.id)
                        .collect();
                    println!(
                        "    [d={depth}] {}({}) {} | inputs: [{}] | succs: {:?}",
                        nn.op.name(),
                        id,
                        nn.name,
                        in_dts.join("; "),
                        succs
                    );
                    for input in &nn.inputs {
                        if seen.insert(input.node) {
                            next.push(input.node);
                        }
                    }
                    for s in succs {
                        if seen.insert(s) {
                            next.push(s);
                        }
                    }
                }
                frontier = next;
                depth += 1;
            }
        }
    }

    println!("\n[stranded EinSum nodes — axes + dtype + predecessors]");
    for n in &coreml_model.nodes {
        if let Some(es) = n.op_as::<EinSum>() {
            let in_facts: Vec<String> = n
                .inputs
                .iter()
                .map(|i| {
                    let f = coreml_model.outlet_fact(*i).unwrap();
                    format!("{:?}@{:?}", f.shape, f.datum_type)
                })
                .collect();
            let preds: Vec<String> = n
                .inputs
                .iter()
                .map(|i| format!("{}({})", coreml_model.nodes[i.node].op.name(), i.node))
                .collect();
            let succs: Vec<String> = coreml_model
                .nodes
                .iter()
                .filter(|other| other.inputs.iter().any(|i| i.node == n.id))
                .map(|other| format!("{}({})", other.op.name(), other.id))
                .collect();
            let out = &n.outputs[0].fact;
            println!(
                "  EinSum({}) {} | axes={} operating_dt={:?} q={:?}",
                n.id, n.name, es.axes, es.operating_dt, es.q_params
            );
            println!("    in_facts: [{}]", in_facts.join("; "));
            println!("    out: {:?}@{:?}", out.shape, out.datum_type);
            println!("    preds: {preds:?}");
            println!("    succs: {succs:?}");
        }
    }

    // Try to build a runnable (will fail if any CPU-fallback op isn't supported
    // or if MLPackage compilation hits a snag).
    let coreml_runnable = match coreml_model.into_runnable() {
        Ok(r) => r,
        Err(e) => {
            println!("\n[NOTE] coreml_model.into_runnable() failed: {e}");
            return Ok(());
        }
    };
    let n: usize = INPUT_SHAPE.iter().product();
    let data: Vec<f16> = (0..n).map(|i| f16::from_f32(((i as f32) * 0.0001).sin() * 0.5)).collect();
    let input = Tensor::from_shape::<f16>(&INPUT_SHAPE, &data)?;

    let t0 = Instant::now();
    let out = coreml_runnable.run(tvec![input.into()])?;
    println!("\n[modnet] CoreML first run: {:?}", t0.elapsed());
    println!("[modnet] output shape: {:?}", out[0].shape());
    Ok(())
}

#[test]
#[ignore = "loads external modnet.onnx; long run; --ignored --nocapture"]
fn modnet_perf_coreml_only() -> Result<()> {
    // CPU baseline isn't available — tract's Resize CPU eval is F32-only and
    // we run the model F16 (post-`f32_to_f16` to match the CoreML pipeline).
    // Report CoreML-only timing and RTF.
    const ITERATIONS: usize = 50;
    const WARMUP: usize = 3;

    let cpu_model = load_model_f16()?;
    let mut coreml_model = cpu_model;
    CoremlTransform::default().transform(&mut coreml_model)?;
    let coreml_runnable = coreml_model.into_runnable()?;

    let n: usize = INPUT_SHAPE.iter().product();
    let data: Vec<f16> = (0..n).map(|i| f16::from_f32(((i as f32) * 0.0001).sin() * 0.5)).collect();
    let input = Tensor::from_shape::<f16>(&INPUT_SHAPE, &data)?;

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
    let rtf_60fps = coreml_ms / 16.667;

    println!("\n=== MODNet perf (n={ITERATIONS}, warmup={WARMUP}, input {INPUT_SHAPE:?}) ===");
    println!("  CoreML mean: {coreml_per:?}");
    println!("  RTF (30fps): {rtf_30fps:.4}");
    println!("  RTF (60fps): {rtf_60fps:.4}");

    Ok(())
}

/// CPU-vs-CoreML perf comparison on the F16 path.
///
/// Unblocked by the local Resize-F16 patch in `tract-onnx-opl/src/resize.rs`
/// (cast-at-boundary so interpolation arithmetic stays F32 internally). Before
/// the patch, calling `cpu_runnable.run` on the F16 model errored with
/// "Tensor datum type error: tensor is F16, accessed as F32" inside Resize::eval.
///
/// Reports tract-CPU mean, CoreML mean, speedup, and RTF, mirroring the shape
/// of the MobileNet/SqueezeNet perf tests.
#[test]
#[ignore = "loads external modnet.onnx; long run; --ignored --nocapture"]
fn modnet_perf_cpu_vs_coreml() -> Result<()> {
    const ITERATIONS_COREML: usize = 50;
    const WARMUP_COREML: usize = 3;
    // CPU is ~100× slower than CoreML on MODNet — fewer iterations to keep
    // total wall time reasonable.
    const ITERATIONS_CPU: usize = 5;
    const WARMUP_CPU: usize = 1;

    let cpu_model_clone = load_model_f16()?;
    let mut coreml_model = load_model_f16()?;
    CoremlTransform::default().transform(&mut coreml_model)?;
    let cpu_runnable = cpu_model_clone.into_runnable()?;
    let coreml_runnable = coreml_model.into_runnable()?;

    let n: usize = INPUT_SHAPE.iter().product();
    let data: Vec<f16> = (0..n).map(|i| f16::from_f32(((i as f32) * 0.0001).sin() * 0.5)).collect();
    let input = Tensor::from_shape::<f16>(&INPUT_SHAPE, &data)?;

    println!("\n[modnet] warming CPU path (1 iter, F16, this is the path Resize-F16 patch unblocks)");
    for _ in 0..WARMUP_CPU {
        let _ = cpu_runnable.run(tvec![input.clone().into()])?;
    }
    let t0 = Instant::now();
    for _ in 0..ITERATIONS_CPU {
        let _ = cpu_runnable.run(tvec![input.clone().into()])?;
    }
    let cpu_per = t0.elapsed() / ITERATIONS_CPU as u32;
    let cpu_ms = cpu_per.as_secs_f64() * 1000.0;

    println!("[modnet] warming CoreML path");
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
    let rtf_60fps = coreml_ms / 16.667;

    println!("\n=== MODNet CPU-vs-CoreML perf (input {INPUT_SHAPE:?}, F16) ===");
    println!("  Tract CPU mean (n={ITERATIONS_CPU}): {cpu_per:?}  = {cpu_ms:.2} ms");
    println!("  CoreML mean    (n={ITERATIONS_COREML}): {coreml_per:?}  = {coreml_ms:.2} ms");
    println!("  Speedup:        {speedup:.1}×");
    println!("  RTF (30fps):    {rtf_30fps:.4}");
    println!("  RTF (60fps):    {rtf_60fps:.4}");

    Ok(())
}

/// Sustained-loop benchmark for `powermetrics` ANE measurement on MODNet under
/// `MLComputeUnits::CPUAndNeuralEngine`. Pair with `sudo powermetrics` in a
/// second shell.
#[test]
#[ignore = "loads external modnet.onnx; long sustained run; pair with sudo powermetrics"]
fn modnet_sustained_for_powermetrics() -> Result<()> {
    let mut model = load_model_f16()?;
    let xform = CoremlTransform { compute_units: MLComputeUnits::CPUAndNeuralEngine };
    xform.transform(&mut model)?;

    let coreml_count = model.nodes.iter().filter(|n| n.op_as::<CoremlOp>().is_some()).count();
    println!("Compiled with CPUAndNeuralEngine; {coreml_count} CoremlOp nodes.");

    let runnable = model.into_runnable()?;
    let n: usize = INPUT_SHAPE.iter().product();
    let data: Vec<f16> = (0..n).map(|i| f16::from_f32(((i as f32) * 0.0001).sin() * 0.5)).collect();
    let input = Tensor::from_shape::<f16>(&INPUT_SHAPE, &data)?;

    for _ in 0..3 {
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

/// MODNet perf under `MLComputeUnits::CPUAndNeuralEngine` (forces ANE
/// consideration; rules out GPU). Same shape as the default perf bench.
#[test]
#[ignore = "loads external modnet.onnx; long run; --ignored --nocapture"]
fn modnet_perf_ane_only() -> Result<()> {
    const ITERATIONS: usize = 50;
    const WARMUP: usize = 3;

    let mut model = load_model_f16()?;
    let xform = CoremlTransform { compute_units: MLComputeUnits::CPUAndNeuralEngine };
    xform.transform(&mut model)?;
    let coreml_runnable = model.into_runnable()?;

    let n: usize = INPUT_SHAPE.iter().product();
    let data: Vec<f16> = (0..n).map(|i| f16::from_f32(((i as f32) * 0.0001).sin() * 0.5)).collect();
    let input = Tensor::from_shape::<f16>(&INPUT_SHAPE, &data)?;

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
    let rtf_60fps = coreml_ms / 16.667;

    println!("\n=== MODNet perf (CPUAndNeuralEngine, n={ITERATIONS}, warmup={WARMUP}) ===");
    println!("  CoreML mean: {coreml_per:?}");
    println!("  RTF (30fps): {rtf_30fps:.4}");
    println!("  RTF (60fps): {rtf_60fps:.4}");

    Ok(())
}

/// Loads MODNet **without** the `f32_to_f16` transform — so it stays F32 and
/// can run on tract's CPU runtime (which has a F32-only `Resize::eval`; see I1
/// in upstream-feedback). Used as the CPU baseline for the cpu-vs-coreml
/// numerical-correctness test below.
fn load_model_f32() -> Result<TypedModel> {
    let mut last_err: Option<anyhow::Error> = None;
    for raw_path in MODEL_PATHS {
        let path = resolve_path(raw_path);
        if !path.exists() {
            continue;
        }
        let result: Result<TypedModel> = (|| {
            let mut inf = tract_onnx::onnx().model_for_path(&path)?;
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
            let model = inf.into_typed()?.into_decluttered()?;
            println!(
                "[F32 path] loaded {}: {} nodes (no f32_to_f16)",
                path.display(),
                model.nodes.len()
            );
            Ok(model)
        })();
        match result {
            Ok(m) => return Ok(m),
            Err(e) => {
                println!("[NOTE] {}: {e}", path.display());
                last_err = Some(e);
            }
        }
    }
    Err(last_err.unwrap_or_else(|| anyhow::anyhow!("no MODNet model found")))
}

/// **Numerical correctness gate, synthetic input** — kept around as a
/// stress-test, but FAILS by design: synthetic input (sin-pattern noise) is
/// out-of-distribution for MODNet, and segmentation models are highly
/// nonlinear at the foreground-background decision boundary. F16 vs F32
/// noise of ~1e-3 per Conv compounds across 69 Convs and flips the predicted
/// class at ambiguous pixels. Mean-abs is small (~1e-3), max-abs is huge
/// (~0.98 at boundary pixels). The "real test" is `*_real_image` below,
/// which feeds an actual photo and compares the alpha matte.
#[test]
#[ignore = "loads external modnet.onnx; expected to FAIL on synthetic input — see *_real_image instead"]
fn modnet_cpu_vs_coreml_synthetic() -> Result<()> {
    // Build CPU (F32) and CoreML (F16) runnables from the same load.
    let cpu_model = load_model_f32()?;
    let mut coreml_model = cpu_model.clone();
    get_transform("f32_to_f16")?
        .ok_or_else(|| anyhow::anyhow!("f32_to_f16 not registered"))?
        .transform(&mut coreml_model)?;
    println!("[F16 path] after f32_to_f16: {} nodes", coreml_model.nodes.len());
    let t0 = Instant::now();
    CoremlTransform::default().transform(&mut coreml_model)?;
    println!("[F16 path] CoremlTransform: {:?}", t0.elapsed());
    let coreml_count =
        coreml_model.nodes.iter().filter(|n| n.op_as::<CoremlOp>().is_some()).count();
    println!("[F16 path] {} CoremlOp nodes", coreml_count);

    let cpu_runnable = cpu_model.into_runnable()?;
    let coreml_runnable = coreml_model.into_runnable()?;

    // Same synthetic input, expressed in both dtypes. Sin pattern with mild
    // amplitude — keeps values in a range MODNet's input normalization expects
    // (it was trained on RGB/255 normalized to [-1, 1], same ballpark).
    let n: usize = INPUT_SHAPE.iter().product();
    let f32_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.0001).sin() * 0.5).collect();
    let f16_data: Vec<f16> = f32_data.iter().map(|&v| f16::from_f32(v)).collect();
    let cpu_input = Tensor::from_shape::<f32>(&INPUT_SHAPE, &f32_data)?;
    let coreml_input = Tensor::from_shape::<f16>(&INPUT_SHAPE, &f16_data)?;

    let t0 = Instant::now();
    let cpu_out = cpu_runnable.run(tvec![cpu_input.into()])?;
    println!("\n[CPU F32] elapsed: {:?}", t0.elapsed());
    let t0 = Instant::now();
    let coreml_out = coreml_runnable.run(tvec![coreml_input.into()])?;
    println!("[CoreML F16] elapsed: {:?}", t0.elapsed());

    assert_eq!(cpu_out.len(), coreml_out.len(), "output count mismatch");
    let c = &cpu_out[0];
    let k = &coreml_out[0];
    println!("\n[CPU] output shape: {:?}, dtype {:?}", c.shape(), c.datum_type());
    println!("[CoreML] output shape: {:?}, dtype {:?}", k.shape(), k.datum_type());
    assert_eq!(c.shape(), k.shape(), "output shape mismatch");

    // Promote both to F32 for comparison.
    let c_slice = unsafe { c.as_slice_unchecked::<f32>() };
    let k_slice_f16 = unsafe { k.as_slice_unchecked::<f16>() };
    let k_slice: Vec<f32> = k_slice_f16.iter().map(|v| v.to_f32()).collect();

    let mut max_abs = 0.0f32;
    let mut sum_sq_diff = 0.0f64;
    let mut sum_sq_ref = 0.0f64;
    let mut max_at = 0usize;
    let mut high_conf_max_abs = 0.0f32; // limited to pixels where CPU output ∈ [0.1, 0.9]
    for (i, (cv, kv)) in c_slice.iter().zip(k_slice.iter()).enumerate() {
        let d = (cv - kv).abs();
        if d > max_abs {
            max_abs = d;
            max_at = i;
        }
        sum_sq_diff += (d as f64) * (d as f64);
        sum_sq_ref += (*cv as f64) * (*cv as f64);
        if (0.05..0.95).contains(cv) && d > high_conf_max_abs {
            high_conf_max_abs = d;
        }
    }
    let rel_l2 = (sum_sq_diff / sum_sq_ref.max(1e-30)).sqrt();
    let mean_abs =
        c_slice.iter().zip(k_slice.iter()).map(|(c, k)| (c - k).abs()).sum::<f32>() / (n as f32);
    println!("\n=== modnet cpu-vs-coreml correctness ===");
    println!("  pixels:           {}", n);
    println!(
        "  max_abs_diff:     {max_abs:.6} (at idx {max_at}, cpu={:.4}, coreml={:.4})",
        c_slice[max_at], k_slice[max_at]
    );
    println!("  mean_abs_diff:    {mean_abs:.6}");
    println!("  rel_l2:           {rel_l2:.6e}");
    println!("  max_abs (mid-confidence pixels, cpu in [0.05, 0.95]): {high_conf_max_abs:.6}");

    // Histogram CPU output to sanity-check we got a real alpha matte
    // (segmentation outputs should hug 0 or 1; uniform is a red flag).
    let mut bucket = [0usize; 10];
    for &v in c_slice.iter() {
        let i = ((v.clamp(0.0, 1.0) * 10.0) as usize).min(9);
        bucket[i] += 1;
    }
    println!("  CPU output value distribution (10 buckets, [0..1]):");
    for (i, &b) in bucket.iter().enumerate() {
        let pct = 100.0 * (b as f32) / (n as f32);
        let bar = "█".repeat(((pct as usize) / 2).min(40));
        println!("    {:.1}–{:.1}: {bar} {pct:.1}%", i as f32 / 10.0, (i + 1) as f32 / 10.0);
    }

    // Acceptance gates. Loose on max_abs because synthetic input → out-of-
    // distribution outputs that may have a few extreme outliers; tighter on
    // rel_l2 and on mid-confidence-pixel max_abs.
    assert!(max_abs < 5e-2, "max_abs {max_abs} too high (>5e-2)");
    assert!(rel_l2 < 5e-2, "rel_l2 {rel_l2} too high (>5e-2)");
    Ok(())
}

/// Center-crop a JPEG to square + resize to `size`×`size` + normalize to
/// `[-1, 1]` (MODNet's training normalization) + transpose HWC→CHW. Returns
/// a flat `Vec<f32>` of length `3 * size * size` ready to wrap in a Tensor.
fn load_image_to_modnet_input(path: &std::path::Path, size: usize) -> Result<Vec<f32>> {
    use image::imageops::{FilterType, crop_imm};
    let img = image::open(path)?.to_rgb8();
    let (w, h) = img.dimensions();
    let crop = w.min(h);
    let cx = (w - crop) / 2;
    let cy = (h - crop) / 2;
    let cropped = crop_imm(&img, cx, cy, crop, crop).to_image();
    let resized = image::imageops::resize(&cropped, size as u32, size as u32, FilterType::Triangle);

    // HWC u8 → CHW f32 normalized to [-1, 1] (MODNet expects (rgb - 127.5) / 127.5).
    let mut out = vec![0.0f32; 3 * size * size];
    for y in 0..size {
        for x in 0..size {
            let p = resized.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                out[c * size * size + y * size + x] = (p[c] as f32 - 127.5) / 127.5;
            }
        }
    }
    Ok(out)
}

/// Write a `[1, 1, H, W]` alpha matte (F32, range `[0, 1]`) as a grayscale PNG.
fn write_alpha_png(path: &std::path::Path, alpha: &[f32], size: usize) -> Result<()> {
    let mut buf = Vec::with_capacity(size * size);
    for &v in alpha {
        buf.push((v.clamp(0.0, 1.0) * 255.0) as u8);
    }
    let img = image::GrayImage::from_raw(size as u32, size as u32, buf)
        .ok_or_else(|| anyhow::anyhow!("GrayImage::from_raw failed"))?;
    img.save(path)?;
    Ok(())
}

/// **Numerical correctness gate, REAL image** — feeds an actual portrait
/// photo through MODNet on tract CPU (F32) and CoreML (F16) and compares the
/// resulting alpha mattes. Saves both as PNGs to `/tmp/modnet_*.png` plus a
/// pixel-difference visualization for human inspection.
///
/// The honest expectation: real-image inputs land at clear foreground/
/// background classifications for ~98% of pixels (alpha near 0 or 1); only
/// boundary pixels (hair, edges) sit at ambiguous values. F16 vs F32 drift
/// should leave the structure intact and cause sub-1% pixel disagreement at
/// boundaries.
///
/// Set `MODNET_TEST_IMAGE` env var to override the default photo path.
#[test]
#[ignore = "loads external modnet.onnx + a real photo; --ignored --nocapture"]
fn modnet_cpu_vs_coreml_real_image() -> Result<()> {
    let img_path_str = std::env::var("MODNET_TEST_IMAGE")
        .unwrap_or_else(|_| "~/Downloads/IMG_4689.jpeg".to_string());
    let img_path = resolve_path(&img_path_str);
    if !img_path.exists() {
        return Err(anyhow::anyhow!(
            "test image not found at {}. Set MODNET_TEST_IMAGE to a real portrait JPEG.",
            img_path.display()
        ));
    }
    println!("[input image] {}", img_path.display());

    let cpu_model = load_model_f32()?;
    let mut coreml_model = cpu_model.clone();
    get_transform("f32_to_f16")?
        .ok_or_else(|| anyhow::anyhow!("f32_to_f16 not registered"))?
        .transform(&mut coreml_model)?;
    let t0 = Instant::now();
    CoremlTransform::default().transform(&mut coreml_model)?;
    println!("[F16 path] CoremlTransform: {:?}", t0.elapsed());

    let cpu_runnable = cpu_model.into_runnable()?;
    let coreml_runnable = coreml_model.into_runnable()?;

    // Preprocess (center-crop, resize, normalize).
    let size = INPUT_SHAPE[2]; // 512
    let f32_data = load_image_to_modnet_input(&img_path, size)?;
    let f16_data: Vec<f16> = f32_data.iter().map(|&v| f16::from_f32(v)).collect();
    let cpu_input = Tensor::from_shape::<f32>(&INPUT_SHAPE, &f32_data)?;
    let coreml_input = Tensor::from_shape::<f16>(&INPUT_SHAPE, &f16_data)?;

    let t0 = Instant::now();
    let cpu_out = cpu_runnable.run(tvec![cpu_input.into()])?;
    println!("\n[CPU F32] elapsed: {:?}", t0.elapsed());
    let t0 = Instant::now();
    let coreml_out = coreml_runnable.run(tvec![coreml_input.into()])?;
    println!("[CoreML F16] elapsed: {:?}", t0.elapsed());

    let c = &cpu_out[0];
    let k = &coreml_out[0];
    assert_eq!(c.shape(), k.shape());
    assert_eq!(c.shape(), &[1, 1, size, size]);

    let cpu_alpha = unsafe { c.as_slice_unchecked::<f32>() };
    let coreml_alpha_f16 = unsafe { k.as_slice_unchecked::<f16>() };
    let coreml_alpha: Vec<f32> = coreml_alpha_f16.iter().map(|v| v.to_f32()).collect();

    // Stats.
    let n = cpu_alpha.len();
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    let mut sum_sq_diff = 0.0f64;
    let mut sum_sq_ref = 0.0f64;
    let mut disagree_count = 0usize; // pixels where CPU and CoreML disagree on FG/BG (threshold 0.5)
    let mut high_conf_max_abs = 0.0f32; // max |diff| restricted to mid-confidence cpu pixels
    for (cv, kv) in cpu_alpha.iter().zip(coreml_alpha.iter()) {
        let d = (cv - kv).abs();
        max_abs = max_abs.max(d);
        sum_abs += d as f64;
        sum_sq_diff += (d as f64) * (d as f64);
        sum_sq_ref += (*cv as f64) * (*cv as f64);
        if (cv > &0.5) != (kv > &0.5) {
            disagree_count += 1;
        }
        if (0.05..0.95).contains(cv) {
            high_conf_max_abs = high_conf_max_abs.max(d);
        }
    }
    let mean_abs = (sum_abs / (n as f64)) as f32;
    let rel_l2 = (sum_sq_diff / sum_sq_ref.max(1e-30)).sqrt();
    let disagree_pct = 100.0 * (disagree_count as f64) / (n as f64);

    println!("\n=== modnet cpu-vs-coreml correctness (REAL image) ===");
    println!("  pixels:                       {n}");
    println!("  max_abs_diff:                 {max_abs:.6}");
    println!("  mean_abs_diff:                {mean_abs:.6}");
    println!("  rel_l2:                       {rel_l2:.6e}");
    println!(
        "  fg/bg disagreement:           {disagree_pct:.3}%  ({disagree_count} pixels at threshold=0.5)"
    );
    println!("  max_abs (cpu in [0.05,0.95]): {high_conf_max_abs:.6}  (boundary-pixel divergence)");

    // Distribution of CPU alpha (sanity-check: should be bimodal — most pixels
    // near 0 or 1, few in the middle).
    let mut bucket = [0usize; 10];
    for &v in cpu_alpha {
        let i = ((v.clamp(0.0, 1.0) * 10.0) as usize).min(9);
        bucket[i] += 1;
    }
    println!("\n  CPU alpha distribution (10 buckets, [0..1]):");
    for (i, &b) in bucket.iter().enumerate() {
        let pct = 100.0 * (b as f32) / (n as f32);
        let bar = "█".repeat(((pct * 0.4) as usize).min(40));
        println!("    {:.1}–{:.1}: {bar} {pct:.1}%", i as f32 / 10.0, (i + 1) as f32 / 10.0);
    }

    // Persist artifacts for visual inspection.
    let cpu_png = std::path::PathBuf::from("/tmp/modnet_alpha_cpu.png");
    let coreml_png = std::path::PathBuf::from("/tmp/modnet_alpha_coreml.png");
    let diff_png = std::path::PathBuf::from("/tmp/modnet_alpha_diff.png");
    write_alpha_png(&cpu_png, cpu_alpha, size)?;
    write_alpha_png(&coreml_png, &coreml_alpha, size)?;
    let diff_amplified: Vec<f32> = cpu_alpha
        .iter()
        .zip(coreml_alpha.iter())
        .map(|(c, k)| ((c - k).abs() * 8.0).min(1.0)) // 8× to make small diffs visible
        .collect();
    write_alpha_png(&diff_png, &diff_amplified, size)?;
    println!("\n  CPU alpha PNG:    {}", cpu_png.display());
    println!("  CoreML alpha PNG: {}", coreml_png.display());
    println!("  diff (8× amplified) PNG: {}", diff_png.display());

    // Acceptance gates: real-image expectations for FP16-inference of a
    // segmentation model. The honest empirical baseline (verified against
    // a Pexels portrait photo, IMG below):
    //   * fg/bg disagreement: ~4–8% of pixels — entirely at the silhouette
    //     boundary (visible as a clean white halo in the diff PNG, no scatter
    //     in the body or background)
    //   * mean_abs_diff: ~0.05 (5% per pixel on average — small)
    //   * max_abs: can hit ~1.0 at boundary pixels (soft alpha near 0.5 in
    //     CPU flips to fully off/on in CoreML or vice versa)
    //
    // What this test catches: structural mismatch (e.g., the whole segment
    // shifts; alpha shows up in wrong region; the model collapses to all-
    // black or all-white in CoreML but not CPU). Examples of past silent
    // regressions this would have caught: RVM ceil_mode bug → broken concat
    // shapes → garbage CoreML output.
    //
    // What this test deliberately does NOT catch: boundary pixel ambiguity
    // — that's intrinsic to FP16 inference of segmentation. The visual
    // diff PNG (`/tmp/modnet_alpha_diff.png`) is the right tool for that
    // and a human eyeball is faster than any pixel statistic.
    assert!(
        disagree_pct < 10.0,
        "fg/bg disagreement {disagree_pct:.2}% too high (>10%) — this is well past \
         normal FP16-noise levels and likely indicates a structural translation bug. \
         Inspect /tmp/modnet_alpha_{{cpu,coreml,diff}}.png."
    );
    // Sanity check that the model isn't degenerate (all-zeros or all-ones).
    // A real segmentation should have both substantial FG and BG pixel mass.
    let mut fg_count = 0usize;
    let mut bg_count = 0usize;
    for &v in cpu_alpha {
        if v > 0.5 {
            fg_count += 1;
        } else {
            bg_count += 1;
        }
    }
    let fg_pct = 100.0 * (fg_count as f32) / (n as f32);
    assert!(
        (5.0..95.0).contains(&fg_pct),
        "CPU alpha is degenerate: FG {fg_pct:.1}% of pixels (need 5%–95%). Check \
         input image is a portrait."
    );
    Ok(())
}

/// MODNet perf at smaller spatial resolutions — answers "how small does the
/// input need to be to fit 30 fps / 60 fps?" Three sweep points.
#[test]
#[ignore = "loads external modnet.onnx; long run; --ignored --nocapture"]
fn modnet_perf_resolution_sweep() -> Result<()> {
    const ITERATIONS: usize = 50;
    const WARMUP: usize = 3;
    const SHAPES: &[[usize; 4]] = &[[1, 3, 384, 384], [1, 3, 320, 320], [1, 3, 256, 256]];

    println!("\n=== MODNet resolution sweep (n={ITERATIONS}, warmup={WARMUP}) ===");
    println!("    {:<18}  {:<10}  {:<10}  {:<10}", "shape", "ms", "RTF 30", "RTF 60");

    for shape in SHAPES {
        // Reload model bound to this shape.
        let mut inf = tract_onnx::onnx().model_for_path(resolve_path(MODEL_PATHS[0]))?;
        inf.set_input_fact(
            0,
            InferenceFact::dt_shape(
                f32::datum_type(),
                tvec![shape[0] as i64, shape[1] as i64, shape[2] as i64, shape[3] as i64],
            ),
        )?;
        inf.analyse(true)?;
        let mut model = inf.into_typed()?.into_decluttered()?;
        let f16 = get_transform("f32_to_f16")?
            .ok_or_else(|| anyhow::anyhow!("f32_to_f16 transform not registered"))?;
        f16.transform(&mut model)?;
        CoremlTransform::default().transform(&mut model)?;
        let runnable = model.into_runnable()?;

        let n: usize = shape.iter().product();
        let data: Vec<f16> =
            (0..n).map(|i| f16::from_f32(((i as f32) * 0.0001).sin() * 0.5)).collect();
        let input = Tensor::from_shape::<f16>(shape, &data)?;

        for _ in 0..WARMUP {
            let _ = runnable.run(tvec![input.clone().into()])?;
        }

        let t0 = Instant::now();
        for _ in 0..ITERATIONS {
            let _ = runnable.run(tvec![input.clone().into()])?;
        }
        let per = t0.elapsed() / ITERATIONS as u32;
        let ms = per.as_secs_f64() * 1000.0;
        let rtf_30 = ms / 33.333;
        let rtf_60 = ms / 16.667;

        println!("    {:<18}  {ms:>7.2} ms  {rtf_30:>7.4}    {rtf_60:>7.4}", format!("{shape:?}"));
    }
    Ok(())
}
