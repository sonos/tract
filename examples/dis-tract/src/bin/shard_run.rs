//! Verify EXO-style per-shard loading with multi-turn decode: run the whole model
//! and a 2-shard chain for N turns (KV looping) and compare the token sequences.
//! Also reports the weight bytes each shard loaded — the memory-split proof.
//!
//!   distract-shard-run <model.nnef.tgz> <n_layers> <cut> <comma,ids> [turns]

use std::collections::HashMap;
use tract_core::prelude::*;
use tract_distributed::llm::load_model;
use tract_distributed::shard_graph::load_shard;

type Runnable = std::sync::Arc<tract_core::model::TypedRunnableModel>;

/// A prepared model plus the names of its ordered inputs/outputs, so a driver can
/// feed by name and loop KV caches (`in_cache_*` <- `out_cache_*`).
struct Runner {
    run: Runnable,
    in_names: Vec<String>,
    in_facts: Vec<TypedFact>,
    out_names: Vec<String>,
}

impl Runner {
    fn new(model: TypedModel, label: &str) -> TractResult<Runner> {
        let opt = model.into_optimized()?;
        let in_names: Vec<String> =
            opt.input_outlets()?.iter().map(|o| opt.node(o.node).name.clone()).collect();
        let in_facts = opt
            .input_outlets()?
            .iter()
            .map(|o| opt.outlet_fact(*o).cloned())
            .collect::<TractResult<Vec<_>>>()?;
        // Derive output names from the actual output nodes so KV feedback matches the
        // model after unfold-kv-cache/declutter, whatever it renamed things to.
        let out_names: Vec<String> = opt
            .output_outlets()?
            .iter()
            .map(|o| {
                opt.outlet_labels.get(o).cloned().unwrap_or_else(|| opt.node(o.node).name.clone())
            })
            .collect();
        if std::env::var("DISTRACT_NAMES").is_ok() {
            eprintln!("[{label}] in : {in_names:?}");
            eprintln!("[{label}] out: {out_names:?}");
        }
        Ok(Runner { run: opt.into_runnable()?, in_names, in_facts, out_names })
    }

    /// Run one turn: read inputs from `state` (empty tensor for an unset cache), then
    /// write every `out_cache_*` back to the matching `in_cache_*` for the next turn.
    /// Returns the first (wire) output.
    fn step(&self, state: &mut HashMap<String, Tensor>) -> TractResult<Tensor> {
        let mut inputs = tvec!();
        for (name, fact) in self.in_names.iter().zip(&self.in_facts) {
            if let Some(t) = state.get(name) {
                inputs.push(t.clone().into_tvalue());
            } else {
                let shape: Vec<usize> = fact
                    .shape
                    .iter()
                    .enumerate()
                    .map(|(ax, d)| d.to_i64().map(|v| v as usize).unwrap_or(usize::from(ax == 0)))
                    .collect();
                inputs.push(Tensor::zero_dt(fact.datum_type, &shape)?.into_tvalue());
            }
        }
        let out = self.run.run(inputs)?;
        // Feed each output cache back to the matching input cache, identified by
        // (key/value, layer) so it works whether the output is named `out_cache_*`
        // (shards) or `in_cache_*_concat` (whole model).
        for (oname, val) in self.out_names.iter().zip(&out) {
            let Some(oid) = cache_id(oname) else { continue };
            if let Some(iname) = self.in_names.iter().find(|n| cache_id(n) == Some(oid)) {
                state.insert(iname.clone(), val.clone().into_tensor());
            }
        }
        Ok(out[0].clone().into_tensor())
    }
}

fn argmax(t: &Tensor) -> i64 {
    let f = t.cast_to::<f32>().unwrap();
    let v = f.view();
    let s = v.as_slice::<f32>().unwrap();
    s.iter()
        .enumerate()
        .fold(
            (0i64, f32::NEG_INFINITY),
            |(bi, bv), (i, &x)| {
                if x > bv { (i as i64, x) } else { (bi, bv) }
            },
        )
        .0
}

fn ids(toks: &[i64]) -> Tensor {
    Tensor::from_shape(&[1, toks.len()], toks).unwrap()
}

/// (is_key, layer) of a cache tensor name like `in_cache_key_5`, `out_cache_value_7`,
/// or `in_cache_key_5_concat` — so an output cache can be matched to its input.
fn cache_id(name: &str) -> Option<(bool, usize)> {
    for (tag, is_key) in [("cache_key_", true), ("cache_value_", false)] {
        if let Some(i) = name.find(tag) {
            let rest = &name[i + tag.len()..];
            let num: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            if let Ok(n) = num.parse::<usize>() {
                return Some((is_key, n));
            }
        }
    }
    None
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let path = std::env::args().nth(1).expect("model.nnef.tgz");
    let n_layers: usize = std::env::args().nth(2).expect("n_layers").parse()?;
    let cut: usize = std::env::args().nth(3).expect("cut").parse()?;
    let prompt: Vec<i64> = std::env::args()
        .nth(4)
        .expect("ids")
        .split(',')
        .filter_map(|x| x.trim().parse().ok())
        .collect();
    let turns: usize = std::env::args().nth(5).and_then(|s| s.parse().ok()).unwrap_or(6);

    // Reference: whole model. Outputs are [logits, out_cache_key_0, out_cache_value_0, …].
    let (full, _n) = load_model(&path)?;
    let reference = Runner::new(full, "ref")?;

    // Shards.
    let (a_model, a_io) = load_shard(&path, 0, cut, n_layers)?;
    let (b_model, b_io) = load_shard(&path, cut, n_layers, n_layers)?;
    println!("shard A [0,{cut})   : {} MiB", a_io.weight_bytes_loaded / 1024 / 1024);
    println!("shard B [{cut},{n_layers}) : {} MiB", b_io.weight_bytes_loaded / 1024 / 1024);
    let resid_name = b_io.residual_in.clone().expect("shard B has a residual input");
    let a = Runner::new(a_model, "A")?;
    let b = Runner::new(b_model, "B")?;

    let (mut ref_state, mut a_state, mut b_state) =
        (HashMap::new(), HashMap::new(), HashMap::new());
    let (mut ref_tok, mut sh_tok) = (0i64, 0i64);
    let mut matches = 0usize;
    println!("\nturn   reference   sharded");
    for turn in 0..turns {
        let ref_in = if turn == 0 { ids(&prompt) } else { ids(&[ref_tok]) };
        let sh_in = if turn == 0 { ids(&prompt) } else { ids(&[sh_tok]) };

        use anyhow::Context;
        ref_state.insert("input_ids".to_string(), ref_in);
        ref_tok = argmax(&reference.step(&mut ref_state).context("reference")?);

        a_state.insert("input_ids".to_string(), sh_in.clone());
        let residual = a.step(&mut a_state).context("shard A")?;
        b_state.insert("input_ids".to_string(), sh_in);
        b_state.insert(resid_name.clone(), residual);
        sh_tok = argmax(&b.step(&mut b_state).context("shard B")?);

        let ok = ref_tok == sh_tok;
        matches += ok as usize;
        println!("{turn:>4}   {ref_tok:>9}   {sh_tok:>7}   {}", if ok { "OK" } else { "MISMATCH" });
    }
    println!("\nparity: {matches}/{turns} tokens match the whole model");
    anyhow::ensure!(matches == turns, "sharded chain diverged");
    Ok(())
}
