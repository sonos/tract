//! Coverage probe: Real-ESRGAN x4 super-resolution ONNX through `CoremlTransform`.
//!
//! Real-ESRGAN is a CNN-based super-resolution model: takes a `[1, 3, H, W]`
//! image and emits `[1, 3, 4*H, 4*W]`. Goal of this probe is to surface what
//! ops are stranded — particularly any "decoder" / upsampling ops like
//! `ConvTranspose`, `PixelShuffle`, `DepthToSpace` — when the model is run
//! through our CoremlTransform.
//!
//! The shipped variant (AXERA-TECH `onnx/realesrgan-x4.onnx`, 64 MB, opset 11,
//! input `[1, 3, 64, 64]` → output `[1, 3, 256, 256]`) actually uses `Resize`
//! pairs for the 2× steps rather than `ConvTranspose` or `PixelShuffle`, which
//! is itself a useful coverage finding.
//!
//! `#[ignore]` by default; loads model from outside the workspace. Run with:
//!
//! ```sh
//! cargo test -p tract-coreml --test realesrgan_e2e -- --ignored --nocapture
//! ```
//!
//! Pure diagnostic: NO CPU-vs-CoreML correctness comparison and NO new
//! translators. Mirrors the structure of `inception_v3_coverage_probe` and
//! `modnet_coverage_probe`.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use half::f16;

use tract_core::internal::*;
use tract_core::ops::cnn::Conv;
use tract_core::transform::{ModelTransform, get_transform};
use tract_onnx::prelude::InferenceModelExt;

use tract_coreml::{CoremlOp, CoremlTransform};

const MODEL_PATH: &str = "~/coding/dfn3-wasm-relaxed-simd/models/realesrgan_x4.onnx";
/// The shipped ONNX has fixed input `[1, 3, 64, 64]` so we just match it; no
/// `set_input_fact` rebinding needed. Output is `[1, 3, 256, 256]` (4× up).
const INPUT_SHAPE: [usize; 4] = [1, 3, 64, 64];

fn resolve_path(path: &str) -> PathBuf {
    if let Some(stripped) = path.strip_prefix("~/") {
        let home = std::env::var("HOME").expect("HOME unset");
        PathBuf::from(home).join(stripped)
    } else {
        PathBuf::from(path)
    }
}

/// Load realesrgan_x4.onnx, declutter, then cast to F16. Mirrors the load
/// shape used by the InceptionV3 / MobileNet canaries: do *not*
/// `into_optimized()` here — that lowers `Conv` to `Im2Col + MatMul` and the
/// translator never sees the high-level Conv op.
fn load_realesrgan_f16() -> Result<TypedModel> {
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
#[ignore = "loads external realesrgan_x4.onnx; run with --ignored --nocapture"]
fn realesrgan_coverage_probe() -> Result<()> {
    let cpu_model = load_realesrgan_f16()?;

    // Op-type histogram, post-declutter+f16. Tells us which ops sit between
    // Convs (= what fusion needs to also translate to grow MLPackages past
    // per-Conv granularity). For a super-resolution model we are especially
    // interested in upsample-flavored ops: ConvTranspose, PixelShuffle (which
    // is itself ONNX `DepthToSpace` mode='CRD' on export), Resize, Upsample.
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
    println!("\n[post-transform op histogram (what's still on CPU)]");
    for (op, count) in &after_hist {
        println!("  {count:4}× {op}");
    }

    // Detail any ConvTranspose / DepthToSpace / Resize / Upsample-like
    // nodes — these are the upsampling primitives Real-ESRGAN-style models
    // commonly use, and the load-bearing question for this probe.
    println!("\n[upsample-flavored stranded nodes — first 5 of each kind]");
    let upsample_keywords = ["ConvTranspose", "DepthToSpace", "PixelShuffle", "Resize", "Upsample"];
    for keyword in upsample_keywords {
        let matches: Vec<&TypedNode> =
            coreml_model.nodes.iter().filter(|n| n.op.name().contains(keyword)).take(5).collect();
        if !matches.is_empty() {
            println!("  -- {keyword}-like ({} shown) --", matches.len());
            for n in matches {
                let in_facts: Vec<String> = n
                    .inputs
                    .iter()
                    .map(|i| {
                        let f = coreml_model.outlet_fact(*i).unwrap();
                        format!("{:?}@{:?}", f.shape, f.datum_type)
                    })
                    .collect();
                let out = &n.outputs[0].fact;
                println!(
                    "    {}({}) {} | in: [{}] | out: {:?}@{:?}",
                    n.op.name(),
                    n.id,
                    n.name,
                    in_facts.join("; "),
                    out.shape,
                    out.datum_type,
                );
            }
        }
    }

    // Detail the *first instance of each stranded op kind* (op-name → first
    // node that has that op-name and isn't already a CoremlOp). Lets us
    // attribute every "skip" to a concrete tract op + datum_type combination.
    println!("\n[first-instance stranded-op detail]");
    let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for n in &coreml_model.nodes {
        if n.op_as::<CoremlOp>().is_some() {
            continue;
        }
        let op_name = n.op.name().to_string();
        if !seen.insert(op_name.clone()) {
            continue;
        }
        let in_facts: Vec<String> = n
            .inputs
            .iter()
            .map(|i| {
                let f = coreml_model.outlet_fact(*i).unwrap();
                format!("{:?}@{:?}", f.shape, f.datum_type)
            })
            .collect();
        let out = &n.outputs[0].fact;
        println!(
            "  {}({}) {} | in: [{}] | out: {:?}@{:?}",
            op_name,
            n.id,
            n.name,
            in_facts.join("; "),
            out.shape,
            out.datum_type,
        );
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
    match coreml_runnable.run(tvec![input.clone().into()]) {
        Ok(coreml_out) => {
            println!("CoreML first run: {:?}", t0.elapsed());
            assert_eq!(coreml_out.len(), 1);
            let k = &coreml_out[0];
            println!("CoreML output shape: {:?}", k.shape());
        }
        Err(e) => {
            println!("\n[NOTE] coreml_runnable.run() failed: {e}");
            println!("[NOTE] still useful coverage signal — see histograms above");
        }
    }

    Ok(())
}

/// CPU-vs-CoreML perf comparison on the F16 path.
///
/// Unblocked by the local Resize-F16 patch in `tract-onnx-opl/src/resize.rs`.
/// Real-ESRGAN's tract-decomposed Resize chain (the 4 stranded upsample ops in
/// the head — see I14) still uses F16 tensors, so pre-patch this errored
/// inside Resize::eval (or — for the `nearest, integer scales` case — inside
/// the Tile chain that the Resize lowers to via declutter).
#[test]
#[ignore = "loads external realesrgan_x4.onnx; long run; --ignored --nocapture"]
fn realesrgan_perf_cpu_vs_coreml() -> Result<()> {
    const ITERATIONS_COREML: usize = 30;
    const WARMUP_COREML: usize = 2;
    const ITERATIONS_CPU: usize = 3;
    const WARMUP_CPU: usize = 1;

    let cpu_model = load_realesrgan_f16()?;
    let mut coreml_model = load_realesrgan_f16()?;
    CoremlTransform::default().transform(&mut coreml_model)?;
    let cpu_runnable = cpu_model.into_runnable()?;
    let coreml_runnable = coreml_model.into_runnable()?;
    let input = make_input()?;

    println!("\n[realesrgan] warming CPU path (F16; this is the path Resize-F16 patch unblocks)");
    for _ in 0..WARMUP_CPU {
        let _ = cpu_runnable.run(tvec![input.clone().into()])?;
    }
    let t0 = Instant::now();
    for _ in 0..ITERATIONS_CPU {
        let _ = cpu_runnable.run(tvec![input.clone().into()])?;
    }
    let cpu_per = t0.elapsed() / ITERATIONS_CPU as u32;
    let cpu_ms = cpu_per.as_secs_f64() * 1000.0;

    println!("[realesrgan] warming CoreML path");
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
    println!("\n=== Real-ESRGAN CPU-vs-CoreML perf (input {INPUT_SHAPE:?}, F16) ===");
    println!("  Tract CPU mean (n={ITERATIONS_CPU}): {cpu_per:?}  = {cpu_ms:.2} ms");
    println!("  CoreML mean    (n={ITERATIONS_COREML}): {coreml_per:?}  = {coreml_ms:.2} ms");
    println!("  Speedup:        {speedup:.1}×");

    Ok(())
}
