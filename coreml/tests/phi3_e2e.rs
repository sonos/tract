//! Phase 4 canary #10 (first LLM): Phi-3-mini-4k-instruct ONNX through
//! `CoremlTransform` — coverage probe only, no correctness check.
//!
//! Phi-3-mini is Microsoft's 3.8B-parameter decoder-only transformer (32
//! layers, 32 heads, head_dim=96, vocab=32064). This is the first LLM
//! canary added to the tract-coreml sweep, intended to surface the gap
//! landscape between what we have today (CNN/transformer-encoder/audio)
//! and what we'd need to translate a decoder-LLM forward pass.
//!
//! ## ONNX variant
//!
//! `microsoft/Phi-3-mini-4k-instruct-onnx` only ships three ONNX bundles
//! (no plain "cpu fp32" anywhere in the repo):
//!
//!   * `cpu_and_mobile/cpu-int4-rtn-block-32` — INT4 RTN quantized (tract
//!     doesn't handle 4-bit quant containers)
//!   * `cuda/cuda-fp16` — FP16, `producer=onnxruntime-genai`. Despite the
//!     "cuda" prefix in the path, the ONNX file itself is just a fused
//!     FP16 export targeting the CUDA EP. We try this one.
//!   * `directml/directml-int4-awq-block-128` — also INT4, also skip.
//!
//! All three are produced by the `onnxruntime-genai` exporter, which
//! aggressively folds the attention block into a single `com.microsoft`
//! custom op (`GroupQueryAttention`) and the norms into
//! `SkipSimplifiedLayerNormalization` (also `com.microsoft`). That is the
//! single biggest finding from this probe — see `notes/probe-phi3-findings.md`.
//!
//! Download recipe (one-time, ~7.6 GB). External-data filenames must be
//! kept verbatim — the `.onnx` file references its `.data` sibling by
//! literal name, so renaming breaks the link.
//! ```sh
//! mkdir -p ~/coding/dfn3-wasm-relaxed-simd/models/phi3 && cd $_
//! BASE=https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx/resolve/main/cuda/cuda-fp16
//! curl -sL -O "$BASE/phi3-mini-4k-instruct-cuda-fp16.onnx"
//! curl -sL -O "$BASE/phi3-mini-4k-instruct-cuda-fp16.onnx.data"
//! ```
//!
//! ```sh
//! cargo test -p tract-coreml --test phi3_e2e -- --ignored --nocapture
//! ```

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;

use tract_core::internal::*;
use tract_core::transform::{ModelTransform, get_transform};
use tract_onnx::prelude::*;

use tract_coreml::{CoremlOp, CoremlTransform};

const MODEL_PATH: &str =
    "~/coding/dfn3-wasm-relaxed-simd/models/phi3/phi3-mini-4k-instruct-cuda-fp16.onnx";

// Phi-3-mini hyperparams (from config.json):
const NUM_LAYERS: usize = 32;
const NUM_HEADS: i64 = 32;
const HEAD_DIM: i64 = 96;

// First-token configuration: batch=1, seq=8, past=0.
const BATCH: i64 = 1;
const SEQ_LEN: i64 = 8;
const TOTAL_SEQ_LEN: i64 = 8; // past + seq when past=0
const PAST_LEN: i64 = 0;

fn resolve_path(path: &str) -> PathBuf {
    if let Some(stripped) = path.strip_prefix("~/") {
        let home = std::env::var("HOME").expect("HOME unset");
        PathBuf::from(home).join(stripped)
    } else {
        PathBuf::from(path)
    }
}

fn load_phi3() -> Result<TypedModel> {
    let path = resolve_path(MODEL_PATH);
    let mut inf = tract_onnx::onnx().model_for_path(&path)?;

    // Slot 0: input_ids [B, S] i64
    inf.set_input_fact(0, InferenceFact::dt_shape(i64::datum_type(), tvec![BATCH, SEQ_LEN]))?;
    // Slot 1: attention_mask [B, T] i64 (T = past + S)
    inf.set_input_fact(1, InferenceFact::dt_shape(i64::datum_type(), tvec![BATCH, TOTAL_SEQ_LEN]))?;
    // Slots 2..=65: past_key_values.{0..31}.{key,value} f16 [B, H, P, D]
    // With PAST_LEN=0 the KV cache tensors are empty (first token).
    for layer in 0..NUM_LAYERS {
        let key_slot = 2 + 2 * layer;
        let val_slot = 3 + 2 * layer;
        inf.set_input_fact(
            key_slot,
            InferenceFact::dt_shape(f16::datum_type(), tvec![BATCH, NUM_HEADS, PAST_LEN, HEAD_DIM]),
        )?;
        inf.set_input_fact(
            val_slot,
            InferenceFact::dt_shape(f16::datum_type(), tvec![BATCH, NUM_HEADS, PAST_LEN, HEAD_DIM]),
        )?;
    }

    // analyse(true) may fail because:
    //   (a) Any com.microsoft custom op (GroupQueryAttention,
    //       SkipSimplifiedLayerNormalization, SimplifiedLayerNormalization)
    //       lands as UnimplementedOp in tract's ONNX importer. Without a
    //       shape-inference rule, facts can't propagate through the network.
    //   (b) Even before the custom ops, tract's INT64 Constant inference may
    //       hit a shape-unification clash on rank-0 vs rank-1 sigil consts
    //       that onnxruntime-genai produces (`/model/constant_nodes/...`).
    // We try strict, then fall back to lax. If even lax fails, we bail with
    // diagnostics — the raw-ONNX op histogram above has already told the
    // coverage story.
    let analyse_strict = inf.analyse(true);
    let model = match analyse_strict {
        Ok(stable) => {
            println!("analyse(true) OK (stable={stable})");
            inf.into_typed()?
        }
        Err(e) => {
            println!("\nanalyse(true) FAILED: {e}");
            println!("Retrying analyse(false) (lax)...");
            match inf.analyse(false) {
                Ok(_) => {
                    println!("analyse(false) OK — proceeding with partial facts");
                    inf.into_typed()?
                }
                Err(e2) => {
                    println!("analyse(false) ALSO FAILED: {e2}");
                    println!("\nDiagnosis: onnxruntime-genai exports use heavy fusion via");
                    println!("  com.microsoft custom ops (GroupQueryAttention,");
                    println!("  Skip/SimplifiedLayerNormalization). These ops, registered");
                    println!("  as UnimplementedOp by tract, lack any shape-inference rule,");
                    println!("  so analyse cannot propagate facts through the network.");
                    return Err(e);
                }
            }
        }
    };
    println!("loaded {}: {} nodes", path.display(), model.nodes.len());

    let model = model.into_decluttered()?;
    println!("after into_decluttered: {} nodes", model.nodes.len());

    // Phi-3-mini cuda-fp16 is already F16; the f32_to_f16 transform is
    // idempotent. We invoke it anyway to match the standard pipeline.
    let mut model = model;
    let f16x = get_transform("f32_to_f16")?
        .ok_or_else(|| anyhow::anyhow!("f32_to_f16 transform not registered"))?;
    f16x.transform(&mut model)?;
    println!("after f32_to_f16: {} nodes", model.nodes.len());
    Ok(model)
}

#[test]
#[ignore = "loads external Phi-3-mini ONNX (~7.6 GB); --ignored --nocapture"]
fn phi3_coverage_probe() -> Result<()> {
    println!("\n========== Phi-3-mini-4k-instruct (cuda-fp16) ==========");

    // Pre-load reconnaissance: read the ONNX file with prost and produce
    // an op-type histogram BEFORE tract analyse, so we still get the
    // ground-truth op landscape even if analyse(true) bails out on
    // UnimplementedOps. We deliberately do this via tract's protobuf
    // model so the test stays self-contained.
    let path = resolve_path(MODEL_PATH);
    let pbmodel = tract_onnx::onnx().proto_model_for_path(&path)?;
    let graph = pbmodel.graph.as_ref().expect("ONNX graph missing");
    println!("\n[raw ONNX] producer={:?} ir_version={}", pbmodel.producer_name, pbmodel.ir_version);
    let mut domain_op_hist: std::collections::BTreeMap<(String, String), usize> =
        Default::default();
    for n in &graph.node {
        let domain = if n.domain.is_empty() { "ai.onnx".to_string() } else { n.domain.clone() };
        *domain_op_hist.entry((domain, n.op_type.clone())).or_default() += 1;
    }
    println!("[raw ONNX op histogram, {} nodes]", graph.node.len());
    for ((dom, op), c) in &domain_op_hist {
        println!("  {c:4}× [{dom}] {op}");
    }

    // Try to load + analyse through tract.
    let cpu_model = match load_phi3() {
        Ok(m) => m,
        Err(e) => {
            println!("\n=== probe terminated early ===");
            println!("load_phi3() failed with: {e:#}");
            println!("\nThe raw-ONNX histogram above already tells the coverage story.");
            return Ok(());
        }
    };

    // Pre-transform op histogram on the typed model.
    let mut op_hist: std::collections::BTreeMap<String, usize> = Default::default();
    for n in &cpu_model.nodes {
        *op_hist.entry(n.op.name().to_string()).or_default() += 1;
    }
    println!("\n[typed-model op histogram, post-declutter+f16, {} nodes]", cpu_model.nodes.len());
    for (op, c) in &op_hist {
        println!("  {c:4}× {op}");
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
    println!("=== after CoremlTransform ===");
    println!("  total nodes: {}", coreml_model.nodes.len());
    println!("  CoremlOps:   {coreml_count}");

    // Per-op-type first-instance dump (mirrors parakeet/nemotron pattern).
    use tract_core::ops::binary::TypedBinOp;
    use tract_core::ops::cast::Cast;
    use tract_core::ops::einsum::EinSum;
    use tract_coreml::ops::binop::{BinOpAnalysis, analyse_binop};
    use tract_coreml::ops::cast::{CastAnalysis, analyse_cast};
    use tract_coreml::ops::einsum::analyse_einsum;
    use tract_coreml::ops::general_matmul::{GeneralMatMulAnalysis, analyse_general_matmul};
    println!("\n[per-op analysis on first stranded instance]");
    let mut seen: std::collections::HashSet<String> = Default::default();
    for n in &coreml_model.nodes {
        let ty = n.op.name().to_string();
        if !seen.insert(ty.clone()) {
            continue;
        }
        let in_shapes: Vec<String> = n
            .inputs
            .iter()
            .map(|i| {
                let f = coreml_model.outlet_fact(*i).unwrap();
                format!("{:?}@{:?}", f.shape, f.datum_type)
            })
            .collect();
        let reason = if n.op_as::<TypedBinOp>().is_some() {
            match analyse_binop(&coreml_model, n) {
                Ok(BinOpAnalysis::Skip(r)) => Some(r),
                Ok(BinOpAnalysis::Translatable(_)) => Some("(would translate)".into()),
                Err(e) => Some(format!("err: {e}")),
            }
        } else if n.op_as::<Cast>().is_some() {
            match analyse_cast(&coreml_model, n) {
                Ok(CastAnalysis::Skip(r)) => Some(r),
                Ok(CastAnalysis::Translatable(_)) => Some("(would translate)".into()),
                Err(e) => Some(format!("err: {e}")),
            }
        } else if n.op_as::<EinSum>().is_some() {
            let einsum_skip = match analyse_einsum(&coreml_model, n) {
                Ok(tract_coreml::ops::conv::ConvAnalysis::Skip(r)) => r,
                _ => "(would translate as conv)".into(),
            };
            let general_skip = match analyse_general_matmul(&coreml_model, n) {
                Ok(GeneralMatMulAnalysis::Skip(r)) => r,
                _ => "(would translate as matmul)".into(),
            };
            Some(format!("einsum: {einsum_skip}; general: {general_skip}"))
        } else {
            None
        };
        if let Some(r) = reason {
            println!("  {ty} ({}): {r}", n.name);
            println!("    inputs: [{}]", in_shapes.join("; "));
        }
    }

    let mut after_hist: std::collections::BTreeMap<String, usize> = Default::default();
    for n in &coreml_model.nodes {
        *after_hist.entry(n.op.name().to_string()).or_default() += 1;
    }
    println!("\n[post-transform op histogram (what's still on CPU)]");
    for (op, c) in &after_hist {
        println!("  {c:4}× {op}");
    }
    Ok(())
}
