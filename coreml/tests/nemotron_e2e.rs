//! Phase 4 canary #9: Nemotron Speech Streaming en 0.6B ASR (NVIDIA, sister to
//! Parakeet, in Sonos CI as `harness/nemotron-speech-streaming-en-0.6b/`).
//!
//! Same four-sub-model RNN-T pattern as Parakeet (preprocessor / encoder /
//! decoder / joint). Encoder is huge (>1 GB) so we skip it like we do for
//! Parakeet — we probe the other three.
//!
//! Symbol naming differs from Parakeet: the NNEF dumps use uppercase, fully
//! qualified names (`BATCH`, `INPUT_SIGNAL__TIME`, `TARGETS__TIME`,
//! `ENCODER_OUTPUTS__TIME`, `DECODER_OUTPUTS__TIME`) instead of Parakeet's
//! single-letter symbols (`B`, `S`, `T`, `U`, `R`).
//!
//!   * **preprocessor** (164 KB) — log-mel feature extraction. Inputs:
//!     `input_signal=[BATCH, INPUT_SIGNAL__TIME] f32`, `length=[BATCH] i64`.
//!     Outputs: `processed_signal=[1, 128, 744] f32`, `processed_length=[1] i64`.
//!   * **decoder** (28 MB) — LSTM-based RNN-T predictor. Inputs:
//!     `targets=[BATCH, TARGETS__TIME] i32`, `states_0/1=[2, BATCH, 640] f32`.
//!     Outputs: `outputs=[1, 640, 1] f32`, `out_states_0/1=[2, 1, 640] f32`.
//!   * **joint** (7 MB) — small dense network (`linear + relu + add +
//!     transpose + unsqueeze`). Inputs: `encoder_outputs=[BATCH, 1024,
//!     ENCODER_OUTPUTS__TIME] f32`, `decoder_outputs=[BATCH, 640,
//!     DECODER_OUTPUTS__TIME] f32`. Output: `outputs=[1, 1, 1, 1025] f32`.
//!
//! Files live in `~/coding/dfn3-wasm-relaxed-simd/models/nemotron/`. Download
//! recipe (one-time):
//! ```sh
//! URL=https://s3.amazonaws.com/tract-ci-builds/tests/asr/613/nvidia--nemotron-speech-streaming-en-0.6b-f32f32
//! mkdir -p ~/coding/dfn3-wasm-relaxed-simd/models/nemotron && cd $_
//! for m in preprocessor decoder joint; do
//!   for ext in nnef.tgz io.npz; do
//!     curl -sL "$URL/nvidia--nemotron-speech-streaming-en-0.6b-f32f32.$m.$ext" -O
//!   done
//! done
//! ```
//!
//! ```sh
//! cargo test -p tract-coreml --test nemotron_e2e -- --ignored --nocapture
//! ```

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;

use tract_core::internal::*;
use tract_core::transform::{ModelTransform, get_transform};
use tract_nnef::prelude::*;

use tract_coreml::{CoremlOp, CoremlTransform};

const MODELS_DIR: &str = "~/coding/dfn3-wasm-relaxed-simd/models/nemotron";
const MODEL_PREFIX: &str = "nvidia--nemotron-speech-streaming-en-0.6b-f32f32";

fn resolve_path(path: &str) -> PathBuf {
    if let Some(stripped) = path.strip_prefix("~/") {
        let home = std::env::var("HOME").expect("HOME unset");
        PathBuf::from(home).join(stripped)
    } else {
        PathBuf::from(path)
    }
}

fn load_submodel(submodel: &str, symbol_subs: &[(&str, i64)]) -> Result<TypedModel> {
    let path = resolve_path(&format!("{MODELS_DIR}/{MODEL_PREFIX}.{submodel}.nnef.tgz"));
    let nnef = tract_nnef::nnef();
    let model = nnef.model_for_path(&path)?;
    println!("loaded {}: {} nodes", path.display(), model.nodes.len());

    // Substitute symbolic shapes (BATCH, INPUT_SIGNAL__TIME, TARGETS__TIME,
    // ENCODER_OUTPUTS__TIME, DECODER_OUTPUTS__TIME) with the concrete values
    // from the matching io.npz bundle. NNEF Nemotron models surface every
    // batch/sequence dim as a symbol; without binding, every translator
    // skips with "data symbolic shape".
    let mut subs = std::collections::HashMap::new();
    for (name, value) in symbol_subs {
        let sym = model.sym(name);
        subs.insert(sym, (*value).to_dim());
    }
    let model = if !subs.is_empty() {
        let m = model.substitute_symbols(&subs)?;
        println!("after substitute_symbols ({:?}): {} nodes", symbol_subs, m.nodes.len());
        m
    } else {
        model
    };

    let model = model.into_decluttered()?;
    println!("after into_decluttered: {} nodes", model.nodes.len());
    let mut model = model;
    let f16x = get_transform("f32_to_f16")?
        .ok_or_else(|| anyhow::anyhow!("f32_to_f16 transform not registered"))?;
    f16x.transform(&mut model)?;
    println!("after f32_to_f16: {} nodes", model.nodes.len());
    Ok(model)
}

fn probe(submodel: &str, symbol_subs: &[(&str, i64)]) -> Result<()> {
    println!("\n========== Nemotron {submodel} ==========");
    let cpu_model = load_submodel(submodel, symbol_subs)?;

    // Pre-transform op histogram.
    let mut op_hist: std::collections::BTreeMap<String, usize> = Default::default();
    for n in &cpu_model.nodes {
        *op_hist.entry(n.op.name().to_string()).or_default() += 1;
    }
    println!("\n[op-type histogram, post-declutter+f16, {} nodes]", cpu_model.nodes.len());
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

    // For each op type, sample one stranded node + show why it didn't translate.
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

#[test]
#[ignore = "loads external Nemotron preprocessor.nnef.tgz (164 KB); --ignored --nocapture"]
fn nemotron_preprocessor_coverage_probe() -> Result<()> {
    // From io.npz: input_signal=[1, 118960] f32, length=[1] i64. The NNEF
    // declares BATCH and INPUT_SIGNAL__TIME as symbols. Concretizing
    // INPUT_SIGNAL__TIME breaks declutter (a shape-of-derived Broadcast
    // expression is left holding the original Sym), so we follow Parakeet's
    // pattern and only bind BATCH; the audio-length axis flows through
    // declutter via shape_of and stays symbolic.
    probe("preprocessor", &[("BATCH", 1)])
}

#[test]
#[ignore = "loads external Nemotron decoder.nnef.tgz (28 MB); --ignored --nocapture"]
fn nemotron_decoder_coverage_probe() -> Result<()> {
    // From io.npz: targets=[1, 1] i32, states_0/1=[2, 1, 640] f32. The
    // NNEF declares only BATCH and TARGETS__TIME as symbols (the state
    // dim 2 and hidden 640 are concrete).
    probe("decoder", &[("BATCH", 1), ("TARGETS__TIME", 1)])
}

#[test]
#[ignore = "loads external Nemotron joint.nnef.tgz (7 MB); --ignored --nocapture"]
fn nemotron_joint_coverage_probe() -> Result<()> {
    // From io.npz: encoder_outputs=[1, 1024, 1] f32, decoder_outputs=[1,
    // 640, 1] f32. Symbols are BATCH, ENCODER_OUTPUTS__TIME,
    // DECODER_OUTPUTS__TIME — bind all three to 1 to match io.npz.
    probe("joint", &[("BATCH", 1), ("ENCODER_OUTPUTS__TIME", 1), ("DECODER_OUTPUTS__TIME", 1)])
}
