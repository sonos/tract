//! M2: split a real decoder LLM that exports its KV cache as external model I/O
//! (`in_cache_*_N` / `out_cache_*_N`). Rather than send the growing cache over
//! the wire, each stage keeps its layers' caches resident and loops them
//! step→step ([`StageState`]); only the residual activation (and token ids /
//! logits at the ends) crosses between machines.

use anyhow::{Context, Result, bail, ensure};
use tract_core::model::extract_subgraph;
use tract_core::prelude::*;

use crate::protocol::{IoSpec, Role};
use crate::runner::StageRunner;

/// Load an NNEF model and apply the transformer detection pass. Coordinator and
/// workers both call this so their `full` model — and therefore the node names
/// `partition_stages` cuts on — are identical.
pub fn load_transformed(path: &str) -> Result<TypedModel> {
    // Load + transform via the SAME high-level path the reference `causal_llm`
    // uses (`tract::nnef().load()` decluttres on load; `Model::transform`
    // decluttres after each pass). The low-level equivalent produces a NaN model
    // on Qwen despite an identical op histogram, so delegate to the API that is
    // known-good, then extract the `TypedModel` for partitioning.
    Ok(load_model(path)?.0)
}

/// Load + transform, also returning `n_regular_outputs`: the number of non-cache
/// outputs (counted BEFORE `unfold-kv-cache` appends the KV outputs). Declutter
/// renames nodes, so cache OUTPUTS cannot be identified by name — they are the
/// outputs at index >= `n_regular_outputs`, the i-th pairing with the i-th cache
/// input. Same convention the reference `causal_llm` relies on.
pub fn load_model(path: &str) -> Result<(TypedModel, usize)> {
    use tract::prelude::*;
    let nnef = tract::nnef()?.with_tract_transformers()?;
    let mut model = nnef.load(path)?;
    model.transform("transformers_detect_all")?;
    let n_regular_outputs = model.output_count()?;
    model.transform("unfold-kv-cache")?;
    Ok((model.into_typed_model(), n_regular_outputs))
}

/// A stage sub-model plus the role of each of its I/O slots.
pub struct StageSpec {
    pub model: TypedModel,
    pub inputs: Vec<IoSpec>,
    pub outputs: Vec<IoSpec>,
}

fn dt_string(dt: DatumType) -> String {
    format!("{dt:?}")
}

fn parse_dt(s: &str) -> Result<DatumType> {
    Ok(match s {
        "F16" => f16::datum_type(),
        "F32" => f32::datum_type(),
        "F64" => f64::datum_type(),
        "I64" => i64::datum_type(),
        "I32" => i32::datum_type(),
        "I8" => i8::datum_type(),
        "U8" => u8::datum_type(),
        "Bool" => bool::datum_type(),
        other => bail!("unsupported cache dtype `{other}`"),
    })
}

/// Empty (P=0) shape for a KV cache `[batch, heads, past, head_dim]`: concrete
/// dims as-is; a symbolic batch (axis 0) → 1; the symbolic past-length axis → 0.
/// Setting a symbolic batch to 0 (the naive `unwrap_or(0)`) yields an empty
/// tensor and NaN attention — Qwen exports the batch dim symbolic, OpenELM
/// concrete, which is why only Qwen hit it.
fn empty_shape(fact: &TypedFact) -> Vec<i64> {
    fact.shape
        .to_tvec()
        .iter()
        .enumerate()
        .map(|(axis, d)| d.to_i64().unwrap_or(if axis == 0 { 1 } else { 0 }))
        .collect()
}

/// Model inputs for a first turn: the prompt token ids plus one empty (P=0) tensor
/// per KV cache input.
pub fn empty_inputs(full: &TypedModel, prompt: &[i64]) -> Result<TVec<TValue>> {
    let mut inputs: TVec<TValue> =
        tvec!(Tensor::from_shape(&[1, prompt.len()], prompt)?.into_tvalue());
    for o in full.input_outlets()?.iter().skip(1) {
        let fact = full.outlet_fact(*o)?;
        let shape: Vec<usize> = empty_shape(fact).iter().map(|d| *d as usize).collect();
        inputs.push(Tensor::zero_dt(fact.datum_type, &shape)?.into_tvalue());
    }
    Ok(inputs)
}

fn layer_index(name: &str) -> Option<usize> {
    name.rsplit('_').next().and_then(|s| s.parse::<usize>().ok())
}

/// Classify a whole model's I/O the same way [`partition_two`] does — token/
/// logits slots are `Wire`, `in_/out_cache_*` slots are `Cache`. Lets the full
/// (unpartitioned) model run through [`StageState`] for a like-for-like
/// reference, with caches seeded empty.
pub fn full_io_roles(model: &TypedModel, n_regular: usize) -> Result<(Vec<IoSpec>, Vec<IoSpec>)> {
    let mut ins = vec![];
    for o in model.input_outlets()? {
        let nm = model.node(o.node).name.clone();
        if nm.contains("cache") {
            let fact = model.outlet_fact(*o)?;
            ins.push(IoSpec {
                role: Role::Cache,
                slot: Some(nm.strip_prefix("in_").unwrap_or(&nm).to_string()),
                dt: dt_string(fact.datum_type),
                shape: empty_shape(fact),
            });
        } else {
            ins.push(IoSpec::wire());
        }
    }
    let mut outs = vec![];
    for (i, _o) in model.output_outlets()?.iter().enumerate() {
        if i >= n_regular {
            outs.push(IoSpec { role: Role::Cache, slot: None, dt: String::new(), shape: vec![] });
        } else {
            outs.push(IoSpec::wire());
        }
    }
    Ok((ins, outs))
}

/// Per-node "cache depth": the highest transformer layer whose `in_cache_*`
/// Source the node transitively depends on, or `None` for nodes computed before
/// any block (embedding, RoPE tables, masks). Declutter renames internal nodes,
/// so this is derived from the graph — only the cache *inputs*, whose names
/// survive as Sources, are read. A node inside block L depends on `in_cache_*_L`,
/// hence depth L; the residual leaving block L also has depth L.
pub fn cache_depths(full: &TypedModel) -> Result<Vec<Option<usize>>> {
    let mut depth: Vec<Option<usize>> = vec![None; full.nodes().len()];
    for &n in full.eval_order()?.iter() {
        let node = full.node(n);
        if node.inputs.is_empty() {
            if node.name.contains("cache") {
                depth[n] = layer_index(&node.name);
            }
            continue;
        }
        depth[n] = node.inputs.iter().filter_map(|i| depth[i.node]).max();
    }
    Ok(depth)
}

/// One standalone module per transformer block: module 0 is prelude+block 0,
/// module `i` is block `i`, the last is block `n-1`+epilogue. This is the
/// static-graph analog of EXO's `layers[start:end]` — a shard is then just a
/// contiguous run of these modules, rather than a bespoke cut of a monolithic
/// graph. Pure reuse: it is [`partition_stages`] cut at every layer boundary.
pub fn split_blocks(
    full: &TypedModel,
    n_layers: usize,
    n_regular: usize,
) -> Result<Vec<StageSpec>> {
    ensure!(n_layers > 0, "model has no transformer blocks");
    let cuts: Vec<usize> = (1..n_layers).collect();
    partition_stages(full, &cuts, n_regular)
}

pub fn partition_stages(
    full: &TypedModel,
    cut_layers: &[usize],
    n_regular: usize,
) -> Result<Vec<StageSpec>> {
    let n_stages = cut_layers.len() + 1;
    let owner = |l: usize| cut_layers.iter().filter(|&&c| c <= l).count();
    let depth = cache_depths(full)?;
    // Where a node is computed: inside block L => stage owner(L); nodes reaching
    // no cache at all (embedding, RoPE tables, masks) are computed in stage 0.
    let node_stage = |n: usize| depth[n].map(&owner).unwrap_or(0);

    let mut token_in = vec![];
    let mut cache_in: Vec<(OutletId, usize, IoSpec)> = vec![];
    for o in full.input_outlets()? {
        let nm = full.node(o.node).name.clone();
        if nm.contains("cache") {
            let layer = layer_index(&nm).context("cache input without layer suffix")?;
            let fact = full.outlet_fact(*o)?;
            cache_in.push((
                *o,
                layer,
                IoSpec {
                    role: Role::Cache,
                    slot: Some(nm.strip_prefix("in_").unwrap_or(&nm).to_string()),
                    dt: dt_string(fact.datum_type),
                    shape: empty_shape(fact),
                },
            ));
        } else {
            token_in.push(*o);
        }
    }

    let mut logits_out = vec![];
    let mut cache_out: Vec<(OutletId, usize, IoSpec)> = vec![];
    for (i, o) in full.output_outlets()?.iter().enumerate() {
        if i < n_regular {
            logits_out.push(*o);
            continue;
        }
        let layer = depth[o.node].context("cache output reaches no cache input")?;
        cache_out.push((
            *o,
            layer,
            IoSpec { role: Role::Cache, slot: None, dt: String::new(), shape: vec![] },
        ));
    }

    // Frontier: every activation produced at or before stage s and consumed after
    // it — the residual stream plus any shared tensor (RoPE, mask, positions).
    // A tensor crossing several stages appears in each frontier it spans, so
    // intermediate stages pass it through. Constants are duplicated into each
    // shard by extract_subgraph, and symbolic dims are recomputed, so neither
    // crosses. This replaces name-based cut points, which declutter breaks.
    let mut frontier: Vec<Vec<OutletId>> = vec![vec![]; n_stages.saturating_sub(1)];
    let mut seen: Vec<std::collections::HashSet<OutletId>> =
        vec![Default::default(); n_stages.saturating_sub(1)];
    for node in full.nodes() {
        let cs = node_stage(node.id);
        for inp in &node.inputs {
            let ps = node_stage(inp.node);
            if ps >= cs {
                continue;
            }
            let prod = full.node(inp.node);
            if prod.inputs.is_empty() && prod.name.contains("cache") {
                continue;
            }
            let fact = full.outlet_fact(*inp)?;
            if fact.konst.is_some() || fact.datum_type == TDim::datum_type() {
                continue;
            }
            for s in ps..cs {
                if seen[s].insert(*inp) {
                    frontier[s].push(*inp);
                }
            }
        }
    }

    let mut stages = Vec::with_capacity(n_stages);
    for s in 0..n_stages {
        let (mut ins, mut in_r) = (vec![], vec![]);
        if s == 0 {
            for o in &token_in {
                ins.push(*o);
                in_r.push(IoSpec::wire());
            }
        } else {
            for o in &frontier[s - 1] {
                ins.push(*o);
                in_r.push(IoSpec::wire());
            }
        }
        for (o, l, spec) in &cache_in {
            if owner(*l) == s {
                ins.push(*o);
                in_r.push(spec.clone());
            }
        }

        let (mut outs, mut out_r) = (vec![], vec![]);
        if s == n_stages - 1 {
            for o in &logits_out {
                outs.push(*o);
                out_r.push(IoSpec::wire());
            }
        } else {
            for o in &frontier[s] {
                outs.push(*o);
                out_r.push(IoSpec::wire());
            }
        }
        for (o, l, spec) in &cache_out {
            if owner(*l) == s {
                outs.push(*o);
                out_r.push(spec.clone());
            }
        }

        stages.push(StageSpec {
            model: extract_subgraph(full, &ins, &outs)?,
            inputs: in_r,
            outputs: out_r,
        });
    }
    Ok(stages)
}

/// A loaded stage that assembles its full input vector from wire tensors + its
/// Empty (P=0) KV tensors for every `Cache`-role input slot.
/// Empty (P=0) KV tensor for each `Cache` input, in input order — so cache slot
/// `i` pairs with the i-th `Cache` input and the i-th `Cache` output (the model
/// declares them in corresponding order, as the reference `causal_llm` relies on).
fn seed_caches(inputs: &[IoSpec]) -> Result<Vec<Tensor>> {
    let mut caches = vec![];
    for s in inputs {
        if s.role == Role::Cache {
            let dt = parse_dt(&s.dt)?;
            let shape: Vec<usize> = s.shape.iter().map(|&d| d.max(0) as usize).collect();
            caches.push(Tensor::zero_dt(dt, &shape)?);
        }
    }
    Ok(caches)
}

/// resident caches, runs, and stores the updated caches — returning only the
/// wire outputs. One instance per worker; reused across decode steps.
pub struct StageState {
    runner: StageRunner,
    inputs: Vec<IoSpec>,
    outputs: Vec<IoSpec>,
    caches: Vec<Tensor>,
}

impl StageState {
    pub fn new(
        model: TypedModel,
        backend: &str,
        inputs: Vec<IoSpec>,
        outputs: Vec<IoSpec>,
    ) -> Result<Self> {
        let caches = seed_caches(&inputs)?;
        Ok(StageState { runner: StageRunner::load(model, backend)?, inputs, outputs, caches })
    }

    /// Clear the resident KV back to empty (P=0), starting a fresh sequence
    /// without reloading weights. Used between prompts on a persistent server.
    pub fn reset(&mut self) -> Result<()> {
        self.caches = seed_caches(&self.inputs)?;
        Ok(())
    }

    /// Run one step. `wire_in` are the Wire-role inputs in slot order.
    pub fn step(&mut self, wire_in: TVec<Tensor>) -> Result<TVec<Tensor>> {
        let mut wire = wire_in.into_iter();
        let mut ci = 0;
        let mut full: TVec<TValue> = tvec!();
        for s in &self.inputs {
            match s.role {
                Role::Wire => full.push(wire.next().context("missing wire input")?.into_tvalue()),
                Role::Cache => {
                    full.push(self.caches[ci].clone().into_tvalue());
                    ci += 1;
                }
            }
        }

        let ishapes: Vec<_> = full.iter().map(|t| t.shape().to_vec()).collect();
        let produced = self.runner.run(full)?;
        if std::env::var("DISTRACT_DEBUG_NAN").is_ok() {
            for (i, v) in produced.iter().enumerate() {
                let nan = if let Ok(f) = v.cast_to::<f32>() {
                    let view = f.view();
                    view.as_slice::<f32>()
                        .map(|s| s.iter().filter(|x| x.is_nan()).count())
                        .unwrap_or(0)
                } else {
                    0
                };
                if nan > 0 {
                    let role =
                        self.outputs.get(i).map(|s| format!("{:?}", s.role)).unwrap_or_default();
                    eprintln!("NaN: out[{i}] {role} {nan} nan; in shapes {ishapes:?}");
                }
            }
        }
        let mut wire_out = tvec!();
        let mut co = 0;
        for (s, v) in self.outputs.iter().zip(produced) {
            let t = v.into_tensor();
            match s.role {
                Role::Wire => wire_out.push(t),
                Role::Cache => {
                    self.caches[co] = t;
                    co += 1;
                }
            }
        }
        Ok(wire_out)
    }
}
