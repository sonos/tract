//! Coverage probe: ViT (Vision Transformer) — the first pure-transformer
//! image canary. Goal is purely diagnostic: surface what ops are stranded
//! when we run a ViT through our `CoremlTransform` so we can prioritize
//! the next translator. No correctness comparison; no perf measurement.
//!
//! Source: HuggingFace `Xenova/vit-base-patch16-224`, input `[1, 3, 224, 224]`.
//!
//! `#[ignore]` by default; loads model from outside the workspace. Run with:
//!
//! ```sh
//! cargo test -p tract-coreml --test vit_e2e -- --ignored --nocapture
//! ```

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;

use tract_core::internal::*;
use tract_core::ops::cnn::Conv;
use tract_core::transform::{ModelTransform, get_transform};
use tract_onnx::prelude::*;

use tract_coreml::{CoremlOp, CoremlTransform};

const MODEL_PATH: &str = "~/coding/dfn3-wasm-relaxed-simd/models/vit_base.onnx";
const INPUT_SHAPE: [i64; 4] = [1, 3, 224, 224];

fn resolve_path(path: &str) -> PathBuf {
    if let Some(stripped) = path.strip_prefix("~/") {
        let home = std::env::var("HOME").expect("HOME unset");
        PathBuf::from(home).join(stripped)
    } else {
        PathBuf::from(path)
    }
}

/// Load vit_base.onnx, declutter, then cast to F16. Mirrors `inception_v3_e2e`
/// — do *not* `into_optimized()` here: that lowers `Conv` to `Im2Col + MatMul`
/// and the translator never sees the high-level `Conv` op.
fn load_vit_f16() -> Result<TypedModel> {
    let path = resolve_path(MODEL_PATH);
    let mut inf = tract_onnx::onnx().model_for_path(&path)?;

    // The Xenova ONNX export keeps batch + num_channels symbolic, so we have
    // to pin the input shape before `analyse`/`into_typed`. ViT-base expects
    // [1, 3, 224, 224].
    inf.set_input_fact(
        0,
        InferenceFact::dt_shape(
            f32::datum_type(),
            tvec![INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2], INPUT_SHAPE[3]],
        ),
    )?;
    inf.analyse(true)?;
    let model = inf.into_typed()?;
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

#[test]
#[ignore = "loads external vit_base.onnx; run with --ignored --nocapture"]
fn vit_coverage_probe() -> Result<()> {
    let cpu_model = load_vit_f16()?;

    // Op-type histogram, post-declutter+f16.
    let mut op_hist: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    for n in &cpu_model.nodes {
        let key = n.op.name().to_string();
        *op_hist.entry(key).or_default() += 1;
    }
    println!("\n[op-type histogram, post-declutter+f16, {} nodes]", cpu_model.nodes.len());
    for (op, count) in &op_hist {
        println!("  {count:4}× {op}");
    }

    // Per-Conv eligibility breakdown (ViT has just the patch-embed Conv,
    // but worth checking that it's translatable).
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

    // Apply CoremlTransform. If it aborts (e.g. internal fusion bug on a
    // particular op pattern), keep going with the pre-transform model so we
    // still get the per-op analysis below — coverage data is the goal.
    let mut coreml_model = cpu_model.clone();
    let t0 = Instant::now();
    let transform_failed: Option<String> =
        match CoremlTransform::default().transform(&mut coreml_model) {
            Ok(()) => {
                println!("\nCoremlTransform: {:?}", t0.elapsed());
                None
            }
            Err(e) => {
                println!("\nCoremlTransform FAILED: {e}");
                println!("  → falling back to pre-transform model for stranded-op analysis");
                Some(e.to_string())
            }
        };

    // If the transform succeeded, we report on the post-transform model;
    // otherwise we fall back to analysing the pre-transform model so we still
    // surface what would have been stranded for the translators we have.
    let analysis_model = if transform_failed.is_some() { &cpu_model } else { &coreml_model };

    let coreml_count =
        coreml_model.nodes.iter().filter(|n| n.op_as::<CoremlOp>().is_some()).count();
    let convs_after = coreml_model.nodes.iter().filter(|n| n.op_as::<Conv>().is_some()).count();
    println!("=== after CoremlTransform ===");
    println!("  total nodes:        {}", coreml_model.nodes.len());
    println!("  Conv nodes left:    {convs_after}  (CPU fallback)");
    println!("  CoremlOp nodes:     {coreml_count}");
    if transform_failed.is_some() {
        println!("  (transform aborted, so above counts reflect partial work)");
    }

    // Post-transform op histogram (what's still on CPU = stranded). When the
    // transform aborted, `analysis_model` is the pre-transform CPU model, so
    // this is "everything" — still useful as the upper bound on stranded ops.
    let mut after_hist: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();
    for n in &analysis_model.nodes {
        let key = n.op.name().to_string();
        *after_hist.entry(key).or_default() += 1;
    }
    let label = if transform_failed.is_some() {
        "[stranded-op histogram (transform aborted; using pre-transform model)]"
    } else {
        "[post-transform op histogram (what's still on CPU = stranded)]"
    };
    println!("\n{label}");
    for (op, count) in &after_hist {
        println!("  {count:4}× {op}");
    }

    // For each op type, sample one stranded node + show why it didn't translate
    // (where we have an `analyse_*` helper to ask). Operate on `analysis_model`
    // so this works whether the transform aborted or completed.
    use tract_core::ops::binary::TypedBinOp;
    use tract_core::ops::cast::Cast;
    use tract_core::ops::einsum::EinSum;
    use tract_core::ops::nn::{Reduce, Softmax};
    use tract_coreml::ops::binop::{BinOpAnalysis, analyse_binop};
    use tract_coreml::ops::cast::{CastAnalysis, analyse_cast};
    use tract_coreml::ops::einsum::analyse_einsum;
    use tract_coreml::ops::general_matmul::{GeneralMatMulAnalysis, analyse_general_matmul};
    use tract_coreml::ops::reduce::{ReduceAnalysis, analyse_reduce};
    use tract_coreml::ops::softmax::{SoftmaxAnalysis, analyse_softmax};
    println!("\n[per-op analysis on first stranded instance]");
    let mut seen: std::collections::HashSet<String> = Default::default();
    for n in &analysis_model.nodes {
        let ty = n.op.name().to_string();
        if !seen.insert(ty.clone()) {
            continue;
        }
        // Skip the CoremlOp itself — it's a successful translation, not stranded.
        if n.op_as::<CoremlOp>().is_some() {
            continue;
        }
        let in_shapes: Vec<String> = n
            .inputs
            .iter()
            .map(|i| {
                let f = analysis_model.outlet_fact(*i).unwrap();
                format!("{:?}@{:?}", f.shape, f.datum_type)
            })
            .collect();
        let reason = if n.op_as::<TypedBinOp>().is_some() {
            match analyse_binop(analysis_model, n) {
                Ok(BinOpAnalysis::Skip(r)) => Some(r),
                Ok(BinOpAnalysis::Translatable(_)) => Some("(would translate)".into()),
                Err(e) => Some(format!("err: {e}")),
            }
        } else if n.op_as::<Cast>().is_some() {
            match analyse_cast(analysis_model, n) {
                Ok(CastAnalysis::Skip(r)) => Some(r),
                Ok(CastAnalysis::Translatable(_)) => Some("(would translate)".into()),
                Err(e) => Some(format!("err: {e}")),
            }
        } else if n.op_as::<EinSum>().is_some() {
            let einsum_skip = match analyse_einsum(analysis_model, n) {
                Ok(tract_coreml::ops::conv::ConvAnalysis::Skip(r)) => r,
                _ => "(would translate as conv)".into(),
            };
            let general_skip = match analyse_general_matmul(analysis_model, n) {
                Ok(GeneralMatMulAnalysis::Skip(r)) => r,
                _ => "(would translate as matmul)".into(),
            };
            Some(format!("einsum: {einsum_skip}; general: {general_skip}"))
        } else if n.op_as::<Reduce>().is_some() {
            match analyse_reduce(analysis_model, n) {
                Ok(ReduceAnalysis::Skip(r)) => Some(r),
                Ok(ReduceAnalysis::Translatable(_)) => Some("(would translate)".into()),
                Err(e) => Some(format!("err: {e}")),
            }
        } else if n.op_as::<Softmax>().is_some() {
            match analyse_softmax(analysis_model, n) {
                Ok(SoftmaxAnalysis::Skip(r)) => Some(r),
                Ok(SoftmaxAnalysis::Translatable(_)) => Some("(would translate)".into()),
                Err(e) => Some(format!("err: {e}")),
            }
        } else {
            None
        };
        if let Some(r) = reason {
            println!("  {ty} ({}): {r}", n.name);
            println!("    inputs: [{}]", in_shapes.join("; "));
        } else {
            println!("  {ty} ({}): (no analyse_* helper)", n.name);
            println!("    inputs: [{}]", in_shapes.join("; "));
        }
    }

    Ok(())
}
