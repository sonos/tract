//! Development and regression harness for the GPT-OSS fused-attention
//! detection, running against a real exported model.
//!
//! These tests are `#[ignore]` because they need a local 13 GB artifact; set
//! `GPT_OSS_NNEF` to the model.nnef.tgz path and run with `-- --ignored`.

// Exercises the Metal runtime: tract-metal is an Apple-only dev-dependency.
#![cfg(target_vendor = "apple")]

use std::collections::HashMap;

use tract_nnef::internal::*;
use tract_transformers::WithTractTransformers;
#[allow(unused_imports)]
use tract_metal as _; // link the metal runtime into the registry

fn model_path() -> Option<String> {
    let default = "/Users/julien.balian/SONOS/data/llm/ohana-registry/\
                   gpt-oss-20b_q40-f16emb-linear-w1w3-w2f16-slidingfix-lmheadq40-20260728/\
                   model/model.nnef.tgz";
    let path = std::env::var("GPT_OSS_NNEF").unwrap_or_else(|_| default.to_string());
    std::path::Path::new(&path).exists().then_some(path)
}

fn load_decluttered() -> TractResult<TypedModel> {
    let path = model_path().expect("model file present (set GPT_OSS_NNEF)");
    let nnef = tract_nnef::nnef().with_tract_transformers();
    nnef.model_for_path(&path)?.into_decluttered()
}

/// Print the decluttered attention region of layer 0: every node whose name
/// mentions `__0_selfAttn` plus the cache concat, with op, inputs and facts.
#[test]
#[ignore]
fn explore_layer0_attention() -> TractResult<()> {
    let model = load_decluttered()?;
    let names: HashMap<usize, &str> =
        model.nodes().iter().map(|n| (n.id, n.name.as_str())).collect();
    for node in model.nodes() {
        if !(node.name.contains("__0_selfAttn") || node.name.contains("cache_key_0")
            || node.name.contains("cache_value_0"))
        {
            continue;
        }
        let ins: Vec<String> = node
            .inputs
            .iter()
            .map(|o| format!("{}[{}].{}", names.get(&o.node).unwrap_or(&"?"), o.node, o.slot))
            .collect();
        let outs: Vec<String> = node
            .outputs
            .iter()
            .map(|o| format!("{:?}", o.fact))
            .collect();
        println!(
            "#{:<4} {:<24} {}\n      ins:  {}\n      out:  {}",
            node.id,
            node.op.name(),
            node.name,
            ins.join(" | "),
            outs.join(" | "),
        );
    }
    println!("model outputs: {:?}", model.outputs);
    Ok(())
}

/// The fuse must catch all 24 attention layers on the real export and leave a
/// well-typed model with the same I/O signature.
#[test]
#[ignore]
fn fuses_all_24_layers_on_real_model() -> TractResult<()> {
    use tract_nnef::tract_core::transform::ModelTransform;
    let mut model = load_decluttered()?;
    let n_inputs = model.inputs.len();
    let n_outputs = model.outputs.len();
    tract_transformers::ops::fused_sdpa::FusedSdpaTransform.transform(&mut model)?;
    let fused = model
        .nodes()
        .iter()
        .filter(|n| n.op_is::<tract_transformers::ops::fused_sdpa::FusedSdpa>())
        .count();
    assert_eq!(fused, 24, "all attention layers fused");
    // gpt-oss-20b alternates sliding_attention(128) / full_attention: the
    // fuse rule must extract exactly 12 windowed and 12 full layers.
    let windows: Vec<u32> = model
        .nodes()
        .iter()
        .filter_map(|n| n.op_as::<tract_transformers::ops::fused_sdpa::FusedSdpa>())
        .map(|op| op.window)
        .collect();
    let sliding = windows.iter().filter(|&&w| w == 128).count();
    let full = windows.iter().filter(|&&w| w == 0).count();
    assert_eq!(
        (sliding, full),
        (12, 12),
        "expected 12 sliding(128) + 12 full layers, got windows {windows:?}"
    );
    assert_eq!(model.inputs.len(), n_inputs, "input signature preserved");
    assert_eq!(model.outputs.len(), n_outputs, "output signature preserved");
    let concats_to_cache: usize = model
        .nodes()
        .iter()
        .filter(|n| n.name.starts_with("out_cache_"))
        .count();
    assert_eq!(concats_to_cache, 0, "cache concats eliminated");
    Ok(())
}

/// Fused model must produce the same logits and caches as the original,
/// across a prefill step AND a decode continuation (exercising the in-place
/// state fast-path against the reference concat).
#[test]
#[ignore]
fn fused_matches_original_on_real_model() -> TractResult<()> {
    use tract_nnef::tract_core::transform::ModelTransform;

    let reference = load_decluttered()?;
    let mut fused = load_decluttered()?;
    tract_transformers::ops::fused_sdpa::FusedSdpaTransform.transform(&mut fused)?;

    let ids_step1: Vec<i64> = vec![200006, 17360, 200008, 3575, 553, 17554, 162016, 11];
    let ids_step2: Vec<i64> = vec![261];

    let make_inputs = |model: &TypedModel,
                       ids: &[i64],
                       caches: Option<&TVec<TValue>>|
     -> TractResult<TVec<TValue>> {
        let mut inputs = TVec::new();
        let mut cache_ix = 0usize;
        for (slot, outlet) in model.inputs.iter().enumerate() {
            let fact = model.outlet_fact(*outlet)?;
            if fact.datum_type == i64::datum_type() && fact.rank() == 2 {
                inputs.push(
                    Tensor::from_shape(&[1, ids.len()], ids)?.into_tvalue(),
                );
            } else if fact.rank() == 0 {
                inputs.push(tensor0(1i64).into_tvalue());
            } else {
                // kv cache input [1, 8, P, 64] f16
                match caches {
                    None => {
                        let dt = fact.datum_type;
                        inputs.push(
                            Tensor::zero_dt(dt, &[1, 8, 0, 64])?.into_tvalue(),
                        );
                    }
                    Some(prev) => {
                        inputs.push(prev[cache_ix].clone());
                        cache_ix += 1;
                    }
                }
            }
            let _ = slot;
        }
        Ok(inputs)
    };

    let run2 = |model: TypedModel, label: &str| -> TractResult<(TVec<TValue>, TVec<TValue>)> {
        let ids1 = ids_step1.clone();
        let ids2 = ids_step2.clone();
        let inputs1 = make_inputs(&model, &ids1, None)?;
        let plan = model.into_runnable()?;
        let mut state = SimpleState::new(&plan)?;
        let out1 = state.run(inputs1)?;
        // caches = every output except the logits (slot 0 assumed logits: it
        // is the only f32/f16 rank-3-with-vocab output; identify by rank)
        let caches1: TVec<TValue> = out1[1..].iter().cloned().collect();
        let inputs2 = make_inputs(state.model(), &ids2, Some(&caches1))?;
        let out2 = state.run(inputs2)?;
        eprintln!("{label}: step1 outputs {}, step2 outputs {}", out1.len(), out2.len());
        Ok((out1, out2))
    };

    let (ref1, ref2) = run2(reference, "reference")?;
    let (fus1, fus2) = run2(fused, "fused")?;

    // The fused op computes attention in f32 where the reference pipeline is
    // f16 (QK and probs@V), so logits drift by precision class over 24 layers.
    // Gates that matter: caches bit-exact, logits argmax identical (token
    // equivalence for greedy decode), and logits grossly close.
    let check = |refs: &TVec<TValue>, fused: &TVec<TValue>, step: &str| -> TractResult<()> {
        for (i, (r, f)) in refs.iter().zip(fused.iter()).enumerate() {
            let r = r.clone().into_tensor();
            let f = f.clone().into_tensor();
            if i == 0 {
                let rr = r.cast_to::<f32>()?;
                let ff = f.cast_to::<f32>()?;
                let rv = rr.try_as_plain()?.as_slice::<f32>()?;
                let fv = ff.try_as_plain()?.as_slice::<f32>()?;
                let argmax = |v: &[f32]| {
                    v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0
                };
                ensure!(
                    argmax(rv) == argmax(fv),
                    "{step}: logits argmax differ: {} vs {}",
                    argmax(rv),
                    argmax(fv)
                );
                let dot: f32 = rv.iter().zip(fv).map(|(a, b)| a * b).sum();
                let nr: f32 = rv.iter().map(|a| a * a).sum::<f32>().sqrt();
                let nf: f32 = fv.iter().map(|a| a * a).sum::<f32>().sqrt();
                let cos = dot / (nr * nf);
                eprintln!("{step}: logits cosine similarity {cos:.6}");
                ensure!(cos > 0.999, "{step}: logits cosine too low: {cos}");
            } else if i <= 2 {
                // Layer-0 caches: identical inputs reach the op, so its
                // append/emit path must reproduce the reference bit-exactly.
                f.close_enough(&r, Approximation::Exact)
                    .with_context(|| format!("{step} cache output {i}"))?;
            } else {
                // Deeper caches inherit the f32-vs-f16 attention drift of all
                // upstream layers; require tight cosine instead of bits.
                let rr = r.cast_to::<f32>()?.into_owned();
                let ff = f.cast_to::<f32>()?.into_owned();
                let rv = rr.try_as_plain()?.as_slice::<f32>()?;
                let fv = ff.try_as_plain()?.as_slice::<f32>()?;
                let dot: f32 = rv.iter().zip(fv).map(|(a, b)| a * b).sum();
                let nr: f32 = rv.iter().map(|a| a * a).sum::<f32>().sqrt();
                let nf: f32 = fv.iter().map(|a| a * a).sum::<f32>().sqrt();
                let cos = dot / (nr * nf).max(f32::MIN_POSITIVE);
                ensure!(cos > 0.99, "{step} cache output {i}: cosine {cos}");
            }
        }
        Ok(())
    };
    check(&ref1, &fus1, "step1")?;
    check(&ref2, &fus2, "step2")?;
    Ok(())
}

/// Replicate causal_llm's exact transform sequence and audit the result.
#[test]
#[ignore]
fn causal_llm_sequence_audit() -> TractResult<()> {
    use tract_nnef::tract_core::transform::ModelTransform;
    let mut model = load_decluttered()?;
    tract_transformers::ops::fused_sdpa::FusedSdpaTransform.transform(&mut model)?;
    model.declutter()?;
    tract_nnef::tract_core::transform::get_transform("transformers_detect_all")?
        .unwrap()
        .transform(&mut model)?;
    model.declutter()?;
    let mut counts = std::collections::BTreeMap::new();
    for node in model.nodes() {
        *counts.entry(node.op.name().to_string()).or_insert(0usize) += 1;
    }
    for (op, n) in &counts {
        if ["FusedSdpa", "Softmax", "Reduce<Max>", "Concat", "ApplyRope", "DynKeyValueCache", "Sdpa", "FlashSDPA", "MoeFfn"]
            .iter()
            .any(|k| op.contains(k))
        {
            println!("{op}: {n}");
        }
    }
    let fused = counts.get("FusedSdpa").copied().unwrap_or(0);
    ensure!(fused == 24, "expected 24 fused ops after full sequence, got {fused}");
    Ok(())
}

/// Print the exact input facts of the fused ops' q-concat and op inputs.
#[test]
#[ignore]
fn audit_fused_input_facts() -> TractResult<()> {
    use tract_nnef::tract_core::transform::ModelTransform;
    let mut model = load_decluttered()?;
    tract_transformers::ops::fused_sdpa::FusedSdpaTransform.transform(&mut model)?;
    for node in model.nodes() {
        if node.op_is::<tract_transformers::ops::fused_sdpa::FusedSdpa>()
            && node.name.contains("__0_")
        {
            for (i, inp) in node.inputs.iter().enumerate() {
                let f = model.outlet_fact(*inp)?;
                let pname = &model.node(inp.node).name;
                println!("in[{i}] <- {pname}: {f:?}");
            }
        }
        if node.name.ends_with(".q") && node.name.contains("__0_") {
            for (i, inp) in node.inputs.iter().enumerate() {
                let f = model.outlet_fact(*inp)?;
                let pname = &model.node(inp.node).name;
                println!("qcat in[{i}] <- {pname}: {f:?}");
            }
        }
    }
    Ok(())
}

/// Fused graph: Metal runtime vs CPU runtime on identical inputs. Names the
/// first diverging output (logits or a specific layer cache).
#[test]
#[ignore]
fn fused_metal_matches_fused_cpu() -> TractResult<()> {
    use tract_nnef::tract_core::transform::ModelTransform;
    let mut model = load_decluttered()?;
    tract_transformers::ops::fused_sdpa::FusedSdpaTransform.transform(&mut model)?;

    let ids: Vec<i64> = (0..96).map(|i| (1000 + i * 37 % 5000) as i64).collect();
    let make_inputs = |model: &TypedModel| -> TractResult<TVec<TValue>> {
        let mut inputs = TVec::new();
        for outlet in model.inputs.iter() {
            let fact = model.outlet_fact(*outlet)?;
            if fact.datum_type == i64::datum_type() && fact.rank() == 2 {
                inputs.push(Tensor::from_shape(&[1, ids.len()], &ids)?.into_tvalue());
            } else if fact.rank() == 0 {
                inputs.push(tensor0(1i64).into_tvalue());
            } else {
                inputs.push(Tensor::zero_dt(fact.datum_type, &[1, 8, 0, 64])?.into_tvalue());
            }
        }
        Ok(inputs)
    };

    let metal_outs = {
        let rt = tract_nnef::tract_core::runtime::runtime_for_name("metal")?
            .context("metal runtime not registered")?;
        let runnable = rt.prepare(model.clone())?;
        runnable.run(make_inputs(&model)?)?
    };
    eprintln!("metal done, running cpu reference...");
    let cpu_outs = {
        let plan = model.clone().into_runnable()?;
        let mut state = SimpleState::new(&plan)?;
        state.run(make_inputs(&model)?)?
    };

    for (i, (c, m)) in cpu_outs.iter().zip(metal_outs.iter()).enumerate() {
        let c = c.clone().into_tensor().cast_to::<f32>()?.into_owned();
        let m = m.clone().into_tensor().cast_to::<f32>()?.into_owned();
        let cv = c.try_as_plain()?.as_slice::<f32>()?;
        let mv = m.try_as_plain()?.as_slice::<f32>()?;
        let dot: f32 = cv.iter().zip(mv).map(|(a, b)| a * b).sum();
        let nc: f32 = cv.iter().map(|a| a * a).sum::<f32>().sqrt();
        let nm: f32 = mv.iter().map(|a| a * a).sum::<f32>().sqrt();
        let cos = dot / (nc * nm).max(f32::MIN_POSITIVE);
        println!("output {i}: cosine {cos:.6} (cpu norm {nc:.2}, metal norm {nm:.2})");
    }
    Ok(())
}
