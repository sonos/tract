//! EXO-style per-shard graph loading: prune the NNEF graph AST to a contiguous
//! layer range [start, end) so a worker materializes only its layers' weights.
//!
//! The raw q40ef16 NNEF is one monolithic graph over all L layers with weights as
//! `variable` consts and explicit per-layer KV cache I/O. This module reads the
//! graph text, keeps only the assignments reachable from the shard's outputs
//! (cutting at its inputs), and reports the `variable` labels the shard needs —
//! so only those `.dat` tensors get read. See notes/distract-m3-shard-loader.md.

use std::collections::{HashMap, HashSet};

use anyhow::Result;
use tract_nnef::ast::{Argument, Assignment, Document, LValue, RValue};

/// Identifier naming in the torch2nnef q40ef16 export.
///
/// The residual add's own name (`hiddenStatesAdd0` vs `Add2`) is a torch2nnef dedup
/// counter that is NOT stable across layers, so the boundary can't be named directly.
/// Instead the residual entering layer N is whatever `__{N}_inputLayernorm_…To0`
/// consumes — [`residual_boundaries`] recovers that from the graph.
fn in_cache(layer: usize) -> [String; 2] {
    [format!("in_cache_key_{layer}"), format!("in_cache_value_{layer}")]
}
fn out_cache(layer: usize) -> [String; 2] {
    [format!("out_cache_key_{layer}"), format!("out_cache_value_{layer}")]
}

/// The result of pruning: which assignments to keep, the shard's I/O, and the
/// `variable` labels (weights) the kept subgraph references.
#[derive(Debug)]
pub struct ShardPlan {
    pub keep: Vec<usize>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub weight_labels: Vec<String>,
    /// The incoming residual identifier that must be turned into an external
    /// (`None` for the first stage, which computes the embedding itself).
    pub residual_in: Option<String>,
}

/// Collect every identifier bound on the left of an assignment.
fn lvalue_idents(l: &LValue, out: &mut Vec<String>) {
    match l {
        LValue::Identifier(id) => out.push(id.0.clone()),
        LValue::Array(v) | LValue::Tuple(v) => v.iter().for_each(|x| lvalue_idents(x, out)),
    }
}

/// Collect every identifier referenced on the right of an assignment.
fn rvalue_idents(r: &RValue, out: &mut Vec<String>) {
    match r {
        RValue::Identifier(id) => out.push(id.0.clone()),
        RValue::Literal(_) => {}
        RValue::Binary(a, _, b) => {
            rvalue_idents(a, out);
            rvalue_idents(b, out);
        }
        RValue::Unary(_, a) => rvalue_idents(a, out),
        RValue::Tuple(v) | RValue::Array(v) => v.iter().for_each(|x| rvalue_idents(x, out)),
        RValue::Subscript(a, s) => {
            rvalue_idents(a, out);
            match &**s {
                tract_nnef::ast::Subscript::Single(x) => rvalue_idents(x, out),
                tract_nnef::ast::Subscript::Range(a, b) => {
                    if let Some(a) = a {
                        rvalue_idents(a, out)
                    }
                    if let Some(b) = b {
                        rvalue_idents(b, out)
                    }
                }
            }
        }
        RValue::Comprehension(c) => {
            for (_, it) in &c.loop_iters {
                rvalue_idents(it, out);
            }
            if let Some(f) = &c.filter {
                rvalue_idents(f, out);
            }
            rvalue_idents(&c.yields, out);
        }
        RValue::IfThenElse(ite) => {
            rvalue_idents(&ite.cond, out);
            rvalue_idents(&ite.then, out);
            rvalue_idents(&ite.otherwise, out);
        }
        RValue::Invocation(inv) => {
            for Argument { rvalue, .. } in &inv.arguments {
                rvalue_idents(rvalue, out);
            }
        }
    }
}

/// Replace every exact `Identifier(from)` reference in an rvalue with `to`.
fn substitute_ident(r: &mut RValue, from: &str, to: &str) {
    match r {
        RValue::Identifier(id) => {
            if id.0 == from {
                id.0 = to.to_string();
            }
        }
        RValue::Literal(_) => {}
        RValue::Binary(a, _, b) => {
            substitute_ident(a, from, to);
            substitute_ident(b, from, to);
        }
        RValue::Unary(_, a) => substitute_ident(a, from, to),
        RValue::Tuple(v) | RValue::Array(v) => {
            v.iter_mut().for_each(|x| substitute_ident(x, from, to))
        }
        RValue::Subscript(a, s) => {
            substitute_ident(a, from, to);
            match &mut **s {
                tract_nnef::ast::Subscript::Single(x) => substitute_ident(x, from, to),
                tract_nnef::ast::Subscript::Range(a, b) => {
                    if let Some(a) = a {
                        substitute_ident(a, from, to)
                    }
                    if let Some(b) = b {
                        substitute_ident(b, from, to)
                    }
                }
            }
        }
        RValue::Comprehension(c) => {
            for (_, it) in &mut c.loop_iters {
                substitute_ident(it, from, to);
            }
            if let Some(f) = &mut c.filter {
                substitute_ident(f, from, to);
            }
            substitute_ident(&mut c.yields, from, to);
        }
        RValue::IfThenElse(ite) => {
            substitute_ident(&mut ite.cond, from, to);
            substitute_ident(&mut ite.then, from, to);
            substitute_ident(&mut ite.otherwise, from, to);
        }
        RValue::Invocation(inv) => {
            for Argument { rvalue, .. } in &mut inv.arguments {
                substitute_ident(rvalue, from, to);
            }
        }
    }
}

/// Find the `variable(label = '...')` labels used anywhere in an rvalue.
fn variable_labels(r: &RValue, out: &mut Vec<String>) {
    if let RValue::Invocation(inv) = r {
        if inv.id.0 == "variable" {
            for Argument { id, rvalue } in &inv.arguments {
                if id.as_ref().map(|i| i.0.as_str()) == Some("label")
                    && let RValue::Literal(tract_nnef::ast::Literal::String(s)) = rvalue
                {
                    out.push(s.clone());
                }
            }
        }
        for Argument { rvalue, .. } in &inv.arguments {
            variable_labels(rvalue, out);
        }
    }
}

/// The residual identifier entering each layer N — the value its input-layernorm
/// cast consumes. `boundary[N]` is the residual leaving layer N-1; `boundary[0]` is
/// the embedding output (the first stage's own input).
pub fn residual_boundaries(doc: &Document) -> HashMap<usize, String> {
    let mut out = HashMap::new();
    for a in &doc.graph_def.body {
        let mut lhs = vec![];
        lvalue_idents(&a.left, &mut lhs);
        for name in lhs {
            // `model_model__{N}_inputLayernorm_hiddenStatesTo0`
            let Some(rest) = name.strip_prefix("model_model__") else { continue };
            let Some((num, tail)) = rest.split_once('_') else { continue };
            if tail != "inputLayernorm_hiddenStatesTo0" {
                continue;
            }
            let Ok(n) = num.parse::<usize>() else { continue };
            let mut deps = vec![];
            rvalue_idents(&a.right, &mut deps);
            if let Some(first) = deps.into_iter().next() {
                out.insert(n, first);
            }
        }
    }
    out
}

/// Prune the graph body to layers [start, end) of an `n_layers`-layer model.
pub fn plan_shard(doc: &Document, start: usize, end: usize, n_layers: usize) -> Result<ShardPlan> {
    anyhow::ensure!(start < end && end <= n_layers, "bad range {start}..{end} of {n_layers}");
    let body = &doc.graph_def.body;
    let boundary = residual_boundaries(doc);

    // identifier -> ALL assignment indices that define it (the graph redefines some
    // identifiers, e.g. a repeated `shape_of`, so a use resolves to the most recent
    // PRIOR definition, not a single global one).
    let mut defs: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, a) in body.iter().enumerate() {
        let mut ids = vec![];
        lvalue_idents(&a.left, &mut ids);
        for id in ids {
            defs.entry(id).or_default().push(i);
        }
    }
    let resolve = |id: &str, use_pos: usize| -> Option<usize> {
        defs.get(id)?.iter().rev().find(|&&j| j < use_pos).copied()
    };

    let is_tail = end == n_layers;
    let boundary_of = |layer: usize| -> Result<String> {
        boundary
            .get(&layer)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("no residual boundary found for layer {layer}"))
    };

    // Shard outputs: the residual entering the NEXT layer (= this shard's last output)
    // or the model logits for the tail stage, plus the owned KV cache outputs.
    let mut outputs = vec![];
    outputs.push(if is_tail { "outputs".to_string() } else { boundary_of(end)? });
    for n in start..end {
        outputs.extend(out_cache(n));
    }

    // The incoming residual's real definition is a non-owned layer: suppress it so
    // reachability doesn't pull that layer in — it becomes an external instead.
    let residual_in = if start > 0 { Some(boundary_of(start)?) } else { None };
    let suppress: HashSet<String> = residual_in.iter().cloned().collect();

    // A non-stage-0 shard reaches the embedding only for its SHAPE (the mask length);
    // the residual input has the same [1,S,hidden] shape, so the gather is aliased to
    // it (in `load_shard`). Here: keep the gather node but don't traverse its deps, so
    // the embed_tokens variable is never reached and never loaded.
    let embed_idx = residual_in
        .is_some()
        .then(|| defs.get("model_model_embedTokens_inputsEmbeds").and_then(|v| v.last().copied()))
        .flatten();

    // Backward reachability from the outputs (used at end-of-graph). Each use carries
    // its position so redefined identifiers resolve to the right prior definition.
    let mut keep: HashSet<usize> = HashSet::new();
    let mut stack: Vec<(String, usize)> = outputs.iter().map(|o| (o.clone(), body.len())).collect();
    while let Some((id, use_pos)) = stack.pop() {
        if suppress.contains(&id) {
            continue;
        }
        if let Some(j) = resolve(&id, use_pos)
            && keep.insert(j)
        {
            if Some(j) == embed_idx {
                continue; // aliased to the residual input; drop its real (embed) deps
            }
            let mut deps = vec![];
            rvalue_idents(&body[j].right, &mut deps);
            for d in deps {
                stack.push((d, j));
            }
        }
    }

    let mut keep: Vec<usize> = keep.into_iter().collect();
    keep.sort_unstable();

    // Weight labels referenced by the kept subgraph.
    let mut weight_labels = vec![];
    for &i in &keep {
        variable_labels(&body[i].right, &mut weight_labels);
    }
    weight_labels.sort_unstable();
    weight_labels.dedup();

    let mut inputs = vec!["input_ids".to_string()];
    if let Some(r) = &residual_in {
        inputs.push(r.clone());
    }
    for n in start..end {
        inputs.extend(in_cache(n));
    }

    Ok(ShardPlan { keep, inputs, outputs, weight_labels, residual_in })
}

/// Extract the layer number from a weight label, if any (globals return None).
pub fn label_layer(label: &str) -> Option<usize> {
    // torch2nnef weights: `model.model.layers.<N>.…` or `model_model__<N>_…`
    if let Some(i) = label.find("layers.") {
        return label[i + 7..].split('.').next().and_then(|s| s.parse().ok());
    }
    if let Some(rest) = label.split("__").nth(1) {
        return rest.split('_').next().and_then(|s| s.parse().ok());
    }
    None
}

/// Keep only the assignments in `plan.keep`, preserving order.
pub fn pruned_body(doc: &Document, plan: &ShardPlan) -> Vec<Assignment> {
    let keep: HashSet<usize> = plan.keep.iter().copied().collect();
    doc.graph_def
        .body
        .iter()
        .enumerate()
        .filter(|(i, _)| keep.contains(i))
        .map(|(_, a)| a.clone())
        .collect()
}

/// The single shape dim of a `variable(label = …substr…, shape = [d])` — used to
/// recover the hidden size for the incoming-residual external.
fn scalar_shape_of(doc: &Document, label_substr: &str) -> Option<usize> {
    fn scan(r: &RValue, sub: &str) -> Option<usize> {
        let RValue::Invocation(inv) = r else { return None };
        if inv.id.0 == "variable" {
            let has = inv.arguments.iter().any(|a| {
                a.id.as_ref().map(|i| i.0.as_str()) == Some("label")
                    && matches!(&a.rvalue, RValue::Literal(tract_nnef::ast::Literal::String(s)) if s.contains(sub))
            });
            if has {
                for a in &inv.arguments {
                    if a.id.as_ref().map(|i| i.0.as_str()) == Some("shape")
                        && let RValue::Array(v) = &a.rvalue
                        && let Some(RValue::Literal(tract_nnef::ast::Literal::Numeric(n))) =
                            v.last()
                    {
                        return n.parse().ok();
                    }
                }
            }
        }
        inv.arguments.iter().find_map(|a| scan(&a.rvalue, sub))
    }
    doc.graph_def.body.iter().find_map(|a| scan(&a.right, label_substr))
}

use std::sync::Arc;
use tract_core::prelude::*;
use tract_nnef::ast::Identifier;

use crate::protocol::IoSpec;

/// Roles of a shard's ordered inputs, so a driver knows which tensor is the token
/// ids, the incoming residual, or a KV cache.
#[derive(Debug, Clone)]
pub struct ShardIo {
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub residual_in: Option<String>,
    pub weight_bytes_loaded: u64,
}

/// Build the per-shard `TypedModel` for layers [start, end), reading ONLY that
/// shard's `.dat` weights from `tgz_path`. Peak weight memory is the shard, not the
/// whole model.
pub fn load_shard(
    tgz_path: &str,
    start: usize,
    end: usize,
    n_layers: usize,
) -> Result<(TypedModel, ShardIo)> {
    use flate2::read::GzDecoder;

    // 1. graph text -> AST -> plan
    let graph_text = read_graph_text(tgz_path)?;
    let mut doc = tract_nnef::ast::parse::parse_document(&graph_text)?;
    let plan = plan_shard(&doc, start, end, n_layers)?;
    let owned: HashSet<String> = plan.weight_labels.iter().cloned().collect();

    // 2. selective tensor read: only this shard's .dat files
    let mut tensors: HashMap<Identifier, Arc<Tensor>> = HashMap::new();
    let mut weight_bytes = 0u64;
    let f = std::fs::File::open(tgz_path)?;
    let mut ar = tar::Archive::new(GzDecoder::new(f));
    for entry in ar.entries()? {
        let mut entry = entry?;
        let path = entry.path()?.to_string_lossy().to_string();
        let Some(label) = path.strip_suffix(".dat") else { continue };
        if !owned.contains(label) {
            continue;
        }
        weight_bytes += entry.size();
        let mut buf = Vec::with_capacity(entry.size() as usize);
        std::io::Read::read_to_end(&mut entry, &mut buf)?;
        let tensor = tract_nnef::tensors::read_tensor(&mut &buf[..])?;
        tensors.insert(Identifier(label.to_string()), Arc::new(tensor));
    }
    anyhow::ensure!(
        tensors.len() == owned.len(),
        "loaded {} of {} shard weights (missing .dat files?)",
        tensors.len(),
        owned.len()
    );

    // 3. rewrite the graph def: pruned body + incoming-residual external + shard I/O
    let mut body = pruned_body(&doc, &plan);
    if let Some(resid) = &plan.residual_in {
        let hidden = scalar_shape_of(&doc, "input_layernorm.weight")
            .ok_or_else(|| anyhow::anyhow!("cannot infer hidden size"))?;
        let ext =
            format!("{resid} = tract_core_external(shape = [1, S, {hidden}], datum_type = 'f16');");
        let mut a = tract_nnef::ast::parse::parse_assignments(&ext)?;
        body.insert(0, a.remove(0));

        // Alias the embedding gather to the residual input (same shape) so the mask's
        // sequence length still resolves without loading embed_tokens.
        for a in &mut body {
            let mut lhs = vec![];
            lvalue_idents(&a.left, &mut lhs);
            if lhs.iter().any(|n| n == "model_model_embedTokens_inputsEmbeds") {
                a.right = RValue::Identifier(Identifier(resid.clone()));
            }
        }
        // The mask/position PAST length derives from layer-0's cache, which a non-first
        // shard doesn't own; repoint it to an owned cache (all caches share a length),
        // then drop the now-unused layer-0 cache external so it isn't a Source/input.
        for a in &mut body {
            substitute_ident(&mut a.right, "in_cache_key_0", &format!("in_cache_key_{start}"));
            substitute_ident(&mut a.right, "in_cache_value_0", &format!("in_cache_value_{start}"));
        }
        body.retain(|a| {
            let mut lhs = vec![];
            lvalue_idents(&a.left, &mut lhs);
            !lhs.iter().any(|n| n == "in_cache_key_0" || n == "in_cache_value_0")
        });
    }
    doc.graph_def.body = body;
    doc.graph_def.parameters = plan.inputs.iter().map(|s| Identifier(s.clone())).collect();
    doc.graph_def.results = plan.outputs.iter().map(|s| Identifier(s.clone())).collect();

    // 4. translate to a TypedModel, then apply the transformer passes
    let proto = tract_nnef::ast::ProtoModel {
        doc,
        tensors,
        quantization: None,
        resources: Default::default(),
    };
    use tract_transformers::WithTractTransformers;
    let nnef = tract_nnef::nnef().with_tract_transformers();
    // Declutter before the transformer passes, as the high-level nnef loader does:
    // the passes' (and the GPU backends') rewrite rules only match decluttered
    // patterns, so a raw translate leaves ops unfused and stranded on the host.
    let mut m: TypedModel = nnef
        .translate(&proto, Default::default())
        .map_err(|(_, e)| anyhow::anyhow!("translate shard: {e:#}"))?
        .into_decluttered()?;

    for name in ["transformers_detect_all", "unfold-kv-cache"] {
        let t = tract_core::transform::get_transform(name)?
            .ok_or_else(|| anyhow::anyhow!("no transform {name}"))?;
        t.transform(&mut m)?;
    }

    let io = ShardIo {
        inputs: plan.inputs,
        outputs: plan.outputs,
        residual_in: plan.residual_in,
        weight_bytes_loaded: weight_bytes,
    };
    Ok((m, io))
}

/// The [start, end) layer range of stage `stage_index` given the plan's cut layers.
pub fn shard_range(cut_layers: &[usize], stage_index: usize, n_layers: usize) -> (usize, usize) {
    let start = if stage_index == 0 { 0 } else { cut_layers[stage_index - 1] };
    let end = cut_layers.get(stage_index).copied().unwrap_or(n_layers);
    (start, end)
}

/// Build the coordinator I/O roles for an OPTIMIZED shard model: `Cache` for the
/// KV `in_/out_cache_*` slots (which loop locally), `Wire` for the token/residual/
/// logits that cross the machine boundary. Must be called on the optimized model —
/// a middle shard drops `input_ids` during optimization.
pub fn shard_io_roles(model: &TypedModel) -> Result<(Vec<IoSpec>, Vec<IoSpec>)> {
    use crate::protocol::Role;
    let mut ins = vec![];
    for o in model.input_outlets()? {
        let nm = model.node(o.node).name.clone();
        if nm.contains("cache") {
            let fact = model.outlet_fact(*o)?;
            let shape: Vec<i64> = fact
                .shape
                .to_tvec()
                .iter()
                .enumerate()
                .map(|(ax, d)| d.to_i64().unwrap_or(if ax == 0 { 1 } else { 0 }))
                .collect();
            ins.push(IoSpec {
                role: Role::Cache,
                slot: Some(nm.strip_prefix("in_").unwrap_or(&nm).to_string()),
                dt: format!("{:?}", fact.datum_type),
                shape,
            });
        } else {
            ins.push(IoSpec::wire());
        }
    }
    let mut outs = vec![];
    for o in model.output_outlets()? {
        let nm =
            model.outlet_labels.get(o).cloned().unwrap_or_else(|| model.node(o.node).name.clone());
        if nm.contains("cache") {
            outs.push(IoSpec { role: Role::Cache, slot: None, dt: String::new(), shape: vec![] });
        } else {
            outs.push(IoSpec::wire());
        }
    }
    Ok((ins, outs))
}

fn read_graph_text(tgz_path: &str) -> Result<String> {
    use flate2::read::GzDecoder;
    let f = std::fs::File::open(tgz_path)?;
    let mut ar = tar::Archive::new(GzDecoder::new(f));
    for entry in ar.entries()? {
        let mut entry = entry?;
        let path = entry.path()?.to_string_lossy().to_string();
        if path.ends_with("graph.nnef") {
            let mut s = String::new();
            std::io::Read::read_to_string(&mut entry, &mut s)?;
            return Ok(s);
        }
    }
    anyhow::bail!("no graph.nnef in {tgz_path}")
}
