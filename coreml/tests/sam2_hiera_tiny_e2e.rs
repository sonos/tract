//! Phase 4 canary #5: SAM 2 Hiera-Tiny image encoder ONNX through
//! `CoremlRuntime`.
//!
//! SAM 2 (Meta, 2024) is a transformer-based image+video segmenter. The
//! `image encoder` sub-model takes a `[1, 3, 1024, 1024]` RGB image and
//! produces multi-scale feature maps that drive the prompt-based mask
//! decoder. The encoder is the heavy compute (~38M params for Hiera-Tiny);
//! the decoder is small + prompt-dependent so we don't benchmark it.
//!
//! Encoder I/O (per `vietanhdev/segment-anything-2-onnx-models`):
//!
//!   inputs:
//!     image           : [1, 3, 1024, 1024]  RGB float32
//!   outputs:
//!     high_res_feats_0 : [1, 32, 256, 256]  stride 4
//!     high_res_feats_1 : [1, 64, 128, 128]  stride 8
//!     image_embed      : [1, 256, 64, 64]   stride 16 (fed to mask decoder)
//!
//! This is the **first transformer-class canary** — it stresses our just-
//! landed transformer-prep batch (Softmax, general MatMul, LayerNorm fold,
//! standalone RmsNorm). Hiera is hierarchical (4 stages of windowed
//! attention) and uses Q/K/V projections + attention + MLP blocks.
//!
//! ```sh
//! cargo test -p tract-coreml --test sam2_hiera_tiny_e2e -- --ignored --nocapture
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

// We use the `.fixed.onnx` variant: the upstream community export
// (vietanhdev/segment-anything-2-onnx-models) ships the encoder with two
// rank-0 value_info entries for `/conv_s0/Conv_output_0` and
// `/conv_s1/Conv_output_0` — ONNX shape inference itself can't deduce them
// (the convs sit at the FPN boundary and the export forgot to record their
// outputs). tract's `analyse(true)` then bails with "Impossible to unify 0
// with 4". The fixed file injects the two known shapes ([1,32,256,256] and
// [1,64,128,128]) as value_info before re-running shape_inference. One-time
// pre-bake — same pattern as RVM's `_baked` workaround.
const MODEL_PATH: &str =
    "~/coding/dfn3-wasm-relaxed-simd/models/sam2_hiera_tiny.encoder.fixed.onnx";
const SRC_SHAPE: [usize; 4] = [1, 3, 1024, 1024];

fn resolve_path(path: &str) -> PathBuf {
    if let Some(stripped) = path.strip_prefix("~/") {
        let home = std::env::var("HOME").expect("HOME unset");
        PathBuf::from(home).join(stripped)
    } else {
        PathBuf::from(path)
    }
}

fn load_sam2_f16() -> Result<TypedModel> {
    let path = resolve_path(MODEL_PATH);
    let mut inf = tract_onnx::onnx().model_for_path(&path)?;

    inf.set_input_fact(
        0,
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
#[ignore = "loads external sam2_hiera_tiny.encoder.onnx; --ignored --nocapture"]
fn sam2_coverage_probe() -> Result<()> {
    let cpu_model = load_sam2_f16()?;

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

    // Diagnose stranded EinSum / general matmul / softmax / layer_norm /
    // rms_norm — these are the new transformer-prep translators most likely
    // to surface gaps on a transformer model.
    use tract_core::ops::einsum::EinSum;
    use tract_core::ops::nn::{RmsNorm, Softmax};
    use tract_coreml::ops::einsum::analyse_einsum;
    use tract_coreml::ops::general_matmul::{GeneralMatMulAnalysis, analyse_general_matmul};
    use tract_coreml::ops::matmul::{MatMulAnalysis, analyse_matmul};
    use tract_coreml::ops::rms_norm::{RmsNormAnalysis, analyse_rms_norm};
    use tract_coreml::ops::softmax::{SoftmaxAnalysis, analyse_softmax};

    println!("\n[stranded EinSum reasons]");
    let mut einsum_axes_hist: std::collections::BTreeMap<String, usize> = Default::default();
    for n in &coreml_model.nodes {
        if let Some(es) = n.op_as::<EinSum>() {
            let einsum_skip = match analyse_einsum(&coreml_model, n)? {
                ConvAnalysis::Translatable(_) => continue,
                ConvAnalysis::Skip(r) => r,
            };
            let matmul_skip = match analyse_matmul(&coreml_model, n)? {
                MatMulAnalysis::Translatable(_) => continue,
                MatMulAnalysis::Skip(r) => r,
            };
            let general_skip = match analyse_general_matmul(&coreml_model, n)? {
                GeneralMatMulAnalysis::Translatable(_) => continue,
                GeneralMatMulAnalysis::Skip(r) => r,
            };
            *einsum_axes_hist.entry(format!("{}", es.axes)).or_default() += 1;
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
            println!("    einsum-skip:  {einsum_skip}");
            println!("    matmul-skip:  {matmul_skip}");
            println!("    general-skip: {general_skip}");
        }
    }
    if !einsum_axes_hist.is_empty() {
        println!("\n[stranded EinSum axes signatures (count × axes)]");
        for (axes, c) in &einsum_axes_hist {
            println!("  {c}× {axes}");
        }
    }

    // Dump stranded BinOp / Cast shapes to understand remaining fragmentation barriers.
    use tract_core::ops::binary::TypedBinOp;
    use tract_core::ops::cast::Cast;
    let mut binop_shape_hist: std::collections::BTreeMap<String, usize> = Default::default();
    let mut cast_shape_hist: std::collections::BTreeMap<String, usize> = Default::default();
    for n in &coreml_model.nodes {
        if n.op_as::<TypedBinOp>().is_some() {
            let in_shapes: Vec<String> = n
                .inputs
                .iter()
                .map(|i| {
                    let f = coreml_model.outlet_fact(*i).unwrap();
                    let r = f.shape.rank();
                    format!("rank{r}")
                })
                .collect();
            *binop_shape_hist.entry(in_shapes.join("+")).or_default() += 1;
        }
        if n.op_as::<Cast>().is_some() {
            let r = coreml_model.outlet_fact(n.inputs[0]).unwrap().shape.rank();
            *cast_shape_hist.entry(format!("rank{r}")).or_default() += 1;
        }
    }
    if !binop_shape_hist.is_empty() {
        println!("\n[stranded BinOp input-rank patterns]");
        for (s, c) in &binop_shape_hist {
            println!("  {c}× {s}");
        }
    }
    if !cast_shape_hist.is_empty() {
        println!("\n[stranded Cast input-rank patterns]");
        for (s, c) in &cast_shape_hist {
            println!("  {c}× {s}");
        }
    }

    // Dump stranded Reshape / MoveAxis details to understand the
    // remaining fragmentation barriers.
    use tract_core::ops::change_axes::AxisOp;
    use tract_coreml::ops::move_axis::{MoveAxisAnalysis, analyse_move_axis};
    use tract_coreml::ops::reshape::{ReshapeAnalysis, analyse_reshape};
    let mut reshape_skip_hist: std::collections::BTreeMap<String, usize> = Default::default();
    let mut move_skip_hist: std::collections::BTreeMap<String, usize> = Default::default();
    for n in &coreml_model.nodes {
        if let Some(ax) = n.op_as::<AxisOp>() {
            match ax {
                AxisOp::Reshape(_, _, _) => {
                    if let ReshapeAnalysis::Skip(r) = analyse_reshape(&coreml_model, n)? {
                        let bucket = r.split(':').next().unwrap_or(&r).to_string();
                        *reshape_skip_hist.entry(bucket).or_default() += 1;
                    }
                }
                AxisOp::Move(_, _) => {
                    if let MoveAxisAnalysis::Skip(r) = analyse_move_axis(&coreml_model, n)? {
                        let bucket = r.split(':').next().unwrap_or(&r).to_string();
                        *move_skip_hist.entry(bucket).or_default() += 1;
                    }
                }
                _ => {}
            }
        }
    }
    if !reshape_skip_hist.is_empty() {
        println!("\n[stranded Reshape skip reasons]");
        for (r, c) in &reshape_skip_hist {
            println!("  {c}× {r}");
        }
    }
    if !move_skip_hist.is_empty() {
        println!("\n[stranded MoveAxis skip reasons]");
        for (r, c) in &move_skip_hist {
            println!("  {c}× {r}");
        }
    }
    println!("\n[stranded Softmax reasons]");
    for n in &coreml_model.nodes {
        if n.op_as::<Softmax>().is_some() {
            match analyse_softmax(&coreml_model, n)? {
                SoftmaxAnalysis::Translatable(_) => continue,
                SoftmaxAnalysis::Skip(r) => println!("  Softmax({}) {}: {r}", n.id, n.name),
            }
        }
    }
    println!("\n[stranded RmsNorm reasons]");
    for n in &coreml_model.nodes {
        if n.op_as::<RmsNorm>().is_some() {
            match analyse_rms_norm(&coreml_model, n)? {
                RmsNormAnalysis::Translatable(_) => continue,
                RmsNormAnalysis::Skip(r) => println!("  RmsNorm({}) {}: {r}", n.id, n.name),
            }
        }
    }

    // Try to build a runnable + run one forward pass.
    let coreml_runnable = match coreml_model.into_runnable() {
        Ok(r) => r,
        Err(e) => {
            println!("\n[NOTE] coreml_model.into_runnable() failed: {e}");
            return Ok(());
        }
    };

    let n: usize = SRC_SHAPE.iter().product();
    let src_data: Vec<f16> =
        (0..n).map(|i| f16::from_f32(((i as f32) * 0.0001).sin() * 0.5)).collect();
    let src = Tensor::from_shape::<f16>(&SRC_SHAPE, &src_data)?;

    let t0 = Instant::now();
    let out = coreml_runnable.run(tvec![src.into()])?;
    println!("\n[sam2] CoreML first run: {:?}", t0.elapsed());
    for (i, t) in out.iter().enumerate() {
        println!("  out[{i}]: shape {:?}", t.shape());
    }
    Ok(())
}

/// Release perf benchmark for SAM 2 Hiera-Tiny image encoder. Single
/// inference on a 1024×1024 image. RTF reference: SAM 2 use cases are
/// typically interactive segmentation where 100–300 ms is acceptable;
/// real-time video matting use cases want <33 ms (30 fps).
#[test]
#[ignore = "loads external sam2 encoder; long run; --ignored --nocapture"]
fn sam2_perf_coreml_only() -> Result<()> {
    const ITERATIONS: usize = 30;
    const WARMUP: usize = 3;

    let mut model = load_sam2_f16()?;
    CoremlTransform::default().transform(&mut model)?;
    let coreml_runnable = model.into_runnable()?;

    let n: usize = SRC_SHAPE.iter().product();
    let src_data: Vec<f16> =
        (0..n).map(|i| f16::from_f32(((i as f32) * 0.0001).sin() * 0.5)).collect();
    let src = Tensor::from_shape::<f16>(&SRC_SHAPE, &src_data)?;

    for _ in 0..WARMUP {
        let _ = coreml_runnable.run(tvec![src.clone().into()])?;
    }

    let t0 = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = coreml_runnable.run(tvec![src.clone().into()])?;
    }
    let coreml_per = t0.elapsed() / ITERATIONS as u32;
    let coreml_ms = coreml_per.as_secs_f64() * 1000.0;
    let rtf_30fps = coreml_ms / 33.333;
    let rtf_60fps = coreml_ms / 16.667;

    println!(
        "\n=== SAM 2 Hiera-Tiny encoder perf (n={ITERATIONS}, warmup={WARMUP}, src {SRC_SHAPE:?}) ==="
    );
    println!("  CoreML mean: {coreml_per:?}");
    println!("  RTF (30fps): {rtf_30fps:.4}");
    println!("  RTF (60fps): {rtf_60fps:.4}");
    Ok(())
}

/// Sustained loop for `powermetrics` ANE measurement.
#[test]
#[ignore = "loads external sam2 encoder; long sustained run; pair with sudo powermetrics"]
fn sam2_sustained_for_powermetrics() -> Result<()> {
    let mut model = load_sam2_f16()?;
    let xform = CoremlTransform { compute_units: MLComputeUnits::CPUAndNeuralEngine };
    xform.transform(&mut model)?;

    let coreml_count = model.nodes.iter().filter(|n| n.op_as::<CoremlOp>().is_some()).count();
    println!("Compiled with CPUAndNeuralEngine; {coreml_count} CoremlOp nodes.");

    let runnable = model.into_runnable()?;
    let n: usize = SRC_SHAPE.iter().product();
    let src_data: Vec<f16> =
        (0..n).map(|i| f16::from_f32(((i as f32) * 0.0001).sin() * 0.5)).collect();
    let src = Tensor::from_shape::<f16>(&SRC_SHAPE, &src_data)?;

    for _ in 0..3 {
        let _ = runnable.run(tvec![src.clone().into()])?;
    }

    let target_secs = 30u64;
    let start = Instant::now();
    let mut iters = 0usize;
    while start.elapsed().as_secs() < target_secs {
        let _ = runnable.run(tvec![src.clone().into()])?;
        iters += 1;
    }
    let elapsed = start.elapsed();
    println!("Ran {iters} iterations in {elapsed:?}  (mean {:?})", elapsed / iters as u32);
    Ok(())
}
