//! Regression harness for the generic fused-attention detection on the real
//! Qwen3.5-35B-A3B export (hybrid: 10 full-attention layers among 30 linear
//! attention layers; head_dim 256, GQA 16q/2kv, no sinks, no window).
//!
//! `#[ignore]` because it needs a local 23 GB artifact; set `QWEN35_NNEF` to
//! the model.nnef.tgz path and run with `-- --ignored`.

use tract_nnef::internal::*;
use tract_transformers::WithTractTransformers;
#[allow(unused_imports)]
use tract_metal as _; // link the metal runtime into the registry

fn model_path() -> Option<String> {
    let default = "/Users/julien.balian/SONOS/data/llm/ohana-qwen35/\
                   qwen35-35b-a3b_q40mm-20260807-195829/model/model.nnef.tgz";
    let path = std::env::var("QWEN35_NNEF").unwrap_or_else(|_| default.to_string());
    std::path::Path::new(&path).exists().then_some(path)
}

fn load_decluttered() -> TractResult<TypedModel> {
    let path = model_path().expect("model file present (set QWEN35_NNEF)");
    let nnef = tract_nnef::nnef().with_tract_transformers();
    nnef.model_for_path(&path)?.into_decluttered()
}

/// The fuse must catch the 10 full-attention layers (and only them), with no
/// sinks and no window, preserving the model I/O signature.
#[test]
#[ignore]
fn fuses_all_10_full_attention_layers() -> TractResult<()> {
    use tract_nnef::tract_core::transform::ModelTransform;
    let mut model = load_decluttered()?;
    let n_inputs = model.inputs.len();
    let n_outputs = model.outputs.len();
    tract_transformers::ops::fused_sdpa::FusedSdpaTransform.transform(&mut model)?;
    let ops: Vec<&tract_transformers::ops::fused_sdpa::FusedSdpa> = model
        .nodes()
        .iter()
        .filter_map(|n| n.op_as::<tract_transformers::ops::fused_sdpa::FusedSdpa>())
        .collect();
    assert_eq!(ops.len(), 10, "all 10 full-attention layers fused");
    assert!(ops.iter().all(|op| !op.has_sinks), "qwen has no attention sinks");
    assert!(ops.iter().all(|op| op.window == 0), "qwen full-attention has no window");
    assert_eq!(model.inputs.len(), n_inputs, "input signature preserved");
    assert_eq!(model.outputs.len(), n_outputs, "output signature preserved");
    let cache_concats: usize = model
        .nodes()
        .iter()
        .filter(|n| {
            n.name.starts_with("out_cache_")
                && n.op_as::<tract_nnef::tract_core::ops::array::TypedConcat>().is_some()
        })
        .count();
    assert_eq!(cache_concats, 0, "cache concats eliminated");
    Ok(())
}

/// Fused model must produce the same logits and caches as the original,
/// across a prefill step AND a decode continuation. Inputs cover the hybrid
/// state mix: growing KV caches (symbolic P) and fixed-size conv/recurrent
/// states.
#[test]
#[ignore]
fn fused_matches_original_on_real_model() -> TractResult<()> {
    use tract_nnef::tract_core::transform::ModelTransform;

    let ids_step1: Vec<i64> = vec![9707, 11, 847, 829, 374, 220, 16, 17];
    let ids_step2: Vec<i64> = vec![18];

    // Resolve an input fact's shape with P = 0 (empty KV past). Fixed-size
    // states have concrete shapes already.
    let make_inputs = |model: &TypedModel,
                       ids: &[i64],
                       states: Option<&TVec<TValue>>|
     -> TractResult<TVec<TValue>> {
        let p = model
            .symbols
            .get("P")
            .context("model has no P symbol")?;
        let values = SymbolValues::default().with(&p, 0);
        let mut inputs = TVec::new();
        let mut state_ix = 0usize;
        for outlet in model.inputs.iter() {
            let fact = model.outlet_fact(*outlet)?;
            if fact.datum_type.is_integer() && fact.rank() == 2 {
                inputs.push(Tensor::from_shape(&[1, ids.len()], ids)?.into_tvalue());
            } else {
                match states {
                    None => {
                        let shape: TVec<usize> = fact
                            .shape
                            .iter()
                            .map(|d| Ok(d.eval(&values).to_usize()?))
                            .collect::<TractResult<_>>()?;
                        inputs.push(Tensor::zero_dt(fact.datum_type, &shape)?.into_tvalue());
                    }
                    Some(prev) => {
                        inputs.push(prev[state_ix].clone());
                        state_ix += 1;
                    }
                }
            }
        }
        Ok(inputs)
    };

    let run2 = |model: TypedModel, label: &str| -> TractResult<(TVec<TValue>, TVec<TValue>)> {
        let inputs1 = make_inputs(&model, &ids_step1, None)?;
        let plan = model.into_runnable()?;
        let mut state = SimpleState::new(&plan)?;
        let out1 = state.run(inputs1)?;
        // model outputs: [logits, states...]; state i feeds input i+1.
        let states1: TVec<TValue> = out1[1..].iter().cloned().collect();
        let inputs2 = make_inputs(state.model(), &ids_step2, Some(&states1))?;
        let out2 = state.run(inputs2)?;
        eprintln!("{label}: step1 outputs {}, step2 outputs {}", out1.len(), out2.len());
        Ok((out1, out2))
    };

    // Load the two model instances sequentially: two 23 GB TypedModels do
    // not fit in RAM side by side.
    let (ref1, ref2) = run2(load_decluttered()?, "reference")?;
    let (fus1, fus2) = {
        let mut fused = load_decluttered()?;
        tract_transformers::ops::fused_sdpa::FusedSdpaTransform.transform(&mut fused)?;
        run2(fused, "fused")?
    };

    // The fused op computes attention in f32 where the reference pipeline is
    // f16 (QK and probs@V), so logits drift by precision class. Gates:
    // logits argmax identical (greedy-token equivalence), logits cosine
    // > 0.999, every state output cosine > 0.99.
    let check = |refs: &TVec<TValue>, fused: &TVec<TValue>, step: &str| -> TractResult<()> {
        for (i, (r, f)) in refs.iter().zip(fused.iter()).enumerate() {
            let r = r.clone().into_tensor().cast_to::<f32>()?.into_owned();
            let f = f.clone().into_tensor().cast_to::<f32>()?.into_owned();
            let rv = r.try_as_plain()?.as_slice::<f32>()?;
            let fv = f.try_as_plain()?.as_slice::<f32>()?;
            let dot: f32 = rv.iter().zip(fv).map(|(a, b)| a * b).sum();
            let nr: f32 = rv.iter().map(|a| a * a).sum::<f32>().sqrt();
            let nf: f32 = fv.iter().map(|a| a * a).sum::<f32>().sqrt();
            let cos = dot / (nr * nf).max(f32::MIN_POSITIVE);
            if i == 0 {
                let argmax = |v: &[f32]| {
                    v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0
                };
                ensure!(
                    argmax(rv) == argmax(fv),
                    "{step}: logits argmax differ: {} vs {}",
                    argmax(rv),
                    argmax(fv)
                );
                eprintln!("{step}: logits cosine {cos:.6}");
                ensure!(cos > 0.999, "{step}: logits cosine too low: {cos}");
            } else {
                ensure!(cos > 0.99, "{step} state output {i}: cosine {cos}");
            }
        }
        Ok(())
    };
    check(&ref1, &fus1, "step1")?;
    check(&ref2, &fus2, "step2")?;
    Ok(())
}
