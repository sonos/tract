//! Phase 4 canary #11 (small standard-ONNX LLM): SmolLM2-135M-Instruct
//! through `CoremlTransform` — coverage probe only, no correctness check.
//!
//! SmolLM2-135M (HuggingFaceTB) is a 135M-parameter Llama-architecture
//! decoder transformer (30 layers, 9 q-heads, 3 kv-heads → true GQA,
//! head_dim=64, vocab=49152). The ONNX file used here is the
//! `optimum-cli`-style export hosted by `onnx-community`, which contains
//! **only standard `ai.onnx@14` ops** — no `com.microsoft` contrib ops
//! like the Microsoft Phi-3 export uses (see `phi3_e2e.rs` /
//! `notes/probe-phi3-findings.md` for that contrast).
//!
//! This is the "what does an unfused decoder LLM actually look like
//! through tract-coreml" probe. The op landscape is wide and granular:
//! 2844 nodes, with explicit RoPE primitives (Cos/Sin/Concat/Neg/Mul),
//! explicit KV-cache concat (Concat is the 2nd most common op), explicit
//! RMSNorm (Pow/ReduceMean/Sqrt/Div/Mul), explicit attention (MatMul,
//! Softmax, MatMul), and per-layer Gather for embedding lookup.
//!
//! Download recipe (one-time, ~539 MB single-file ONNX, no .data sibling).
//! ```sh
//! mkdir -p ~/coding/dfn3-wasm-relaxed-simd/models/smollm2_135m && cd $_
//! curl -sL -o model.onnx \
//!   https://huggingface.co/onnx-community/SmolLM2-135M-Instruct-ONNX/resolve/main/onnx/model.onnx
//! ```
//!
//! ```sh
//! cargo test -p tract-coreml --test smollm2_e2e -- --ignored --nocapture
//! ```

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;

use tract_core::internal::*;
use tract_core::transform::{ModelTransform, get_transform};
use tract_onnx::prelude::*;

use tract_coreml::{CoremlOp, CoremlTransform};

const MODEL_PATH: &str = "~/coding/dfn3-wasm-relaxed-simd/models/smollm2_135m/model.onnx";

// SmolLM2-135M hyperparams (from config.json):
//   hidden_size=576, num_hidden_layers=30,
//   num_attention_heads=9, num_key_value_heads=3,
//   head_dim = hidden_size / num_attention_heads = 64,
//   vocab_size=49152, rope_theta=100000.
const NUM_LAYERS: usize = 30;
const NUM_KV_HEADS: i64 = 3;
const HEAD_DIM: i64 = 64;

// First-token configuration: batch=1, seq=8, past=0.
// (TOTAL_SEQ_LEN = past + seq = 8.)
const BATCH: i64 = 1;
const SEQ_LEN: i64 = 8;
const TOTAL_SEQ_LEN: i64 = 8;
const PAST_LEN: i64 = 0;

fn resolve_path(path: &str) -> PathBuf {
    if let Some(stripped) = path.strip_prefix("~/") {
        let home = std::env::var("HOME").expect("HOME unset");
        PathBuf::from(home).join(stripped)
    } else {
        PathBuf::from(path)
    }
}

fn load_smollm2() -> Result<TypedModel> {
    let path = resolve_path(MODEL_PATH);
    // The Optimum-exported ONNX has 2844 value_info entries that all use
    // the symbolic dim names `batch_size`, `sequence_length`, and
    // `past_sequence_length + 1`. When tract reads those into per-node
    // facts, they unify against the concrete shapes we forward-derive
    // from set_input_fact and the analyse() rules clash on the very first
    // Gather (e.g. `Sym(batch_size)` vs `Val(1)`). Disable value_info
    // ingestion so analyse() runs purely on input-fact propagation.
    let mut inf = tract_onnx::onnx().with_ignore_value_info(true).model_for_path(&path)?;

    // Inspect declared input slots & bind concrete shapes for each. The
    // graph has 63 inputs total: 3 textual (input_ids, attention_mask,
    // position_ids), then 30 × (past_key, past_value).
    //
    // Slot 0  : input_ids       [B, S]                i64
    // Slot 1  : attention_mask  [B, T]    (T=past+S)  i64
    // Slot 2  : position_ids    [B, S]                i64
    // Slot 3+2k: past_key_values.k.key   [B, KV_H, P, D]  f32
    // Slot 4+2k: past_key_values.k.value [B, KV_H, P, D]  f32
    let n_inputs = inf.inputs.len();
    println!(
        "ONNX declares {n_inputs} inputs (expected: 3 + 2 × {NUM_LAYERS} = {})",
        3 + 2 * NUM_LAYERS
    );

    // Bind ALL inputs by enumerating + matching name/index conventions.
    for slot in 0..n_inputs {
        let name = inf.node(inf.inputs[slot].node).name.clone();
        let fact = if slot == 0 || (name.starts_with("input_ids")) {
            InferenceFact::dt_shape(i64::datum_type(), tvec![BATCH, SEQ_LEN])
        } else if slot == 1 || name.starts_with("attention_mask") {
            InferenceFact::dt_shape(i64::datum_type(), tvec![BATCH, TOTAL_SEQ_LEN])
        } else if slot == 2 || name.starts_with("position_ids") {
            InferenceFact::dt_shape(i64::datum_type(), tvec![BATCH, SEQ_LEN])
        } else {
            // past_key_values.{k}.{key,value}
            InferenceFact::dt_shape(
                f32::datum_type(),
                tvec![BATCH, NUM_KV_HEADS, PAST_LEN, HEAD_DIM],
            )
        };
        inf.set_input_fact(slot, fact)?;
    }
    println!("set_input_fact done for all {n_inputs} inputs");

    // Reset every declared output fact to wildcard so analyse() doesn't
    // unify our concrete forward-derived shapes against the symbolic
    // (`batch_size`, `sequence_length`, `past_sequence_length + 1`)
    // back-propagated shapes that the ONNX file declares for `logits`
    // and the 60 `present.k.{key,value}` outputs. Without this, the
    // first Gather ("/model/embed_tokens/Gather") fails to unify
    // [1, 8, 576] with [batch_size, sequence_length, 576].
    let n_outputs = inf.outputs.len();
    for slot in 0..n_outputs {
        inf.set_output_fact(slot, InferenceFact::default())?;
    }
    println!("set_output_fact (wildcard) done for all {n_outputs} outputs");

    // analyse(true) is the strict path. Optimum-exported ONNX should
    // propagate cleanly because every op is standard ai.onnx@14 and tract
    // has shape-inference rules for all of them.
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
                    return Err(e);
                }
            }
        }
    };
    println!("loaded {}: {} nodes (typed)", path.display(), model.nodes.len());

    let model = match model.into_decluttered() {
        Ok(m) => {
            println!("after into_decluttered: {} nodes", m.nodes.len());
            m
        }
        Err(e) => {
            println!("\ninto_decluttered FAILED: {e}");
            return Err(e);
        }
    };

    let mut model = model;
    let f16x = get_transform("f32_to_f16")?
        .ok_or_else(|| anyhow::anyhow!("f32_to_f16 transform not registered"))?;
    match f16x.transform(&mut model) {
        Ok(()) => println!("after f32_to_f16: {} nodes", model.nodes.len()),
        Err(e) => {
            println!("\nf32_to_f16 FAILED: {e}");
            return Err(e);
        }
    }
    Ok(model)
}

#[test]
#[ignore = "loads external SmolLM2-135M ONNX (~539 MB); --ignored --nocapture"]
fn smollm2_coverage_probe() -> Result<()> {
    println!("\n========== SmolLM2-135M-Instruct (FP32 ONNX, optimum-style) ==========");

    // Pre-load reconnaissance: read the ONNX file with prost and dump
    // the raw-ONNX op histogram by (domain, op_type) — so we still have
    // the op landscape even if analyse() bails out.
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

    // Try to load + analyse + declutter + f16 through tract.
    let cpu_model = match load_smollm2() {
        Ok(m) => m,
        Err(e) => {
            println!("\n=== probe terminated early ===");
            println!("load_smollm2() failed with: {e:#}");
            return Ok(());
        }
    };

    // Pre-transform (i.e. post-declutter+f16) op histogram on typed model.
    let mut op_hist: std::collections::BTreeMap<String, usize> = Default::default();
    for n in &cpu_model.nodes {
        *op_hist.entry(n.op.name().to_string()).or_default() += 1;
    }
    println!("\n[typed-model op histogram, post-declutter+f16, {} nodes]", cpu_model.nodes.len());
    let mut sorted: Vec<_> = op_hist.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
    for (op, c) in &sorted {
        println!("  {c:4}× {op}");
    }

    // Apply CoremlTransform. We wrap in catch_unwind because, on the
    // first attempt against this graph, the transform panics with
    // "model input unmapped" at coreml/src/transform.rs:218 — the
    // mapping built during translation does not contain entries for some
    // model-input outlets (likely the 60 PAST_LEN=0 past_key_values
    // tensors that get eliminated upstream of the transform). We want
    // the per-op skip-reason dump even when the panic occurs, so we
    // catch and continue.
    let mut coreml_model = cpu_model.clone();
    let t0 = Instant::now();
    let mut transform_outcome = "ok";
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        CoremlTransform::default().transform(&mut coreml_model)
    }));
    match result {
        Ok(Ok(())) => println!("\nCoremlTransform: {:?}", t0.elapsed()),
        Ok(Err(e)) => {
            println!("\nCoremlTransform FAILED (Err): {e}");
            transform_outcome = "err";
        }
        Err(payload) => {
            let msg = if let Some(s) = payload.downcast_ref::<&'static str>() {
                (*s).to_string()
            } else if let Some(s) = payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "<non-string panic payload>".to_string()
            };
            println!("\nCoremlTransform PANICKED: {msg}");
            println!("(elapsed before panic: {:?})", t0.elapsed());
            transform_outcome = "panic";
        }
    }

    let coreml_count =
        coreml_model.nodes.iter().filter(|n| n.op_as::<CoremlOp>().is_some()).count();
    println!("=== after CoremlTransform (outcome={transform_outcome}) ===");
    println!("  total nodes: {}", coreml_model.nodes.len());
    println!("  CoremlOps:   {coreml_count}");

    // Per-op-type first-instance dump (mirrors parakeet/nemotron/phi3 pattern).
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
    let mut sorted: Vec<_> = after_hist.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
    for (op, c) in &sorted {
        println!("  {c:4}× {op}");
    }
    Ok(())
}
