//! Phase 4 canary #7: Parakeet TDT 0.6B v3 ASR (NVIDIA, in Sonos CI).
//!
//! Parakeet is in `tract/harness/parakeet-tdt-600m-v3/` and is part of the
//! standard `tract-metal` / `tract-cuda` CI sweep — running it through
//! `tract-coreml` is the apples-to-apples benchmark for backend parity.
//!
//! Four sub-models (we probe three; encoder is 2.3 GB so deferred):
//!
//!   * **preprocessor** (174 KB) — hand-rolled log-mel feature extraction.
//!     No STFT op (implemented via matmul + per-sample math). Many small
//!     ops; will likely have several stranded gaps (`select`, `dyn_slice`,
//!     comparisons `eq/ge/lt`, `log`, `tract_core_bitnot`).
//!   * **decoder** (47 MB) — LSTM-based RNN-T predictor. Has `lstm`,
//!     `lstm_scan_loop`, `tract_core_scan`. Same architectural issue as
//!     DFN3's GRU — the recurrent body stays on CPU until we add a scan-body
//!     recursive transform.
//!   * **joint** (25 MB) — small dense network: `linear + relu + add +
//!     transpose + unsqueeze`. Should be fully translatable with what we
//!     have. The "easy win" / proof of viability.
//!
//! Files live in `~/coding/dfn3-wasm-relaxed-simd/models/parakeet/`. Download
//! recipe (one-time):
//! ```sh
//! URL=https://s3.amazonaws.com/tract-ci-builds/tests/asr/608/nvidia--parakeet-tdt-0.6b-v3-f32f32
//! mkdir -p ~/coding/dfn3-wasm-relaxed-simd/models/parakeet && cd $_
//! for m in preprocessor decoder joint; do
//!   for ext in nnef.tgz io.npz; do
//!     curl -sL "$URL/nvidia--parakeet-tdt-0.6b-v3-f32f32.$m.$ext" -O
//!   done
//! done
//! ```
//!
//! ```sh
//! cargo test -p tract-coreml --test parakeet_e2e -- --ignored --nocapture
//! ```

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;

use tract_core::internal::*;
use tract_core::transform::{ModelTransform, get_transform};
use tract_nnef::prelude::*;

use tract_coreml::{CoremlOp, CoremlTransform};

const MODELS_DIR: &str = "~/coding/dfn3-wasm-relaxed-simd/models/parakeet";
const MODEL_PREFIX: &str = "nvidia--parakeet-tdt-0.6b-v3-f32f32";

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

    // Substitute symbolic shapes (B, R, U, S, ...) with the concrete values
    // from the matching io.npz bundle. NNEF Parakeet models surface every
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
    println!("\n========== Parakeet {submodel} ==========");
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
#[ignore = "loads external Parakeet preprocessor.nnef.tgz (174 KB); --ignored --nocapture"]
fn parakeet_preprocessor_coverage_probe() -> Result<()> {
    // From io.npz: input_signal=[1, 118960], length=[1]. Symbols are batch
    // (B) and signal length (S). Try B=1; the audio length is what tract
    // probably calls S or T in this graph.
    probe("preprocessor", &[("B", 1), ("S", 118960), ("T", 118960)])
}

#[test]
#[ignore = "loads external Parakeet decoder.nnef.tgz (47 MB); --ignored --nocapture"]
fn parakeet_decoder_coverage_probe() -> Result<()> {
    // From io.npz: targets=[1,1] int32, states_0/1=[2,1,640]. Symbols
    // typically B=1 (batch), U=1, T=1 (sequence positions).
    probe("decoder", &[("B", 1), ("U", 1), ("T", 1)])
}

#[test]
#[ignore = "loads external Parakeet joint.nnef.tgz (25 MB); --ignored --nocapture"]
fn parakeet_joint_coverage_probe() -> Result<()> {
    // From io.npz: encoder_outputs=[1, 1024, 1], decoder_outputs=[1, 640, 1].
    // The seen symbolic shape was [B, R, U, 640]. B=1 (batch), R=1
    // (encoder seq position — typical for the joint computation per token),
    // U=1 (decoder seq position).
    probe("joint", &[("B", 1), ("R", 1), ("U", 1)])
}
