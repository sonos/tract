//! Blockify — typed-model rewrite that factors block-diagonal structure
//! into the graph topology, so the result has a single streaming axis
//! everywhere and pulsifies under v1's existing machinery.
//!
//! Phase B POC scope: recognise a single block-diagonal pattern.
//!
//!   EinSum([a, b]) producing scores[T, T]
//!   → Mul(scores, mask) where mask has uniform_tdim `(coord_a/k == coord_b/k)`
//!   → Reduce<Sum> on one of the streaming axes
//!
//! Rewrite: introduce a chunk symbol, factor the streaming dim into
//! [chunks, k], rewrite the einsum subscript with a chunk batch axis,
//! delete the mask construction, adjust downstream axis indices,
//! flatten back at the model boundary.

use crate::internal::*;
use tract_core::axes::AxesMapping;
use tract_core::ops::binary::TypedBinOp;
use tract_core::ops::change_axes::AxisOp;
use tract_core::ops::einsum::EinSum;
use tract_core::ops::math::Mul;
use tract_core::ops::nn::{Reduce, Reducer};

/// Result of running Blockify: the (possibly-rewritten) model, plus the
/// streaming symbol and pulse value to use downstream.  Blockify substitutes
/// the user's stream symbol `T` with `P*S` in the rewritten subgraph;
/// downstream pulsification must use `S` and a translated pulse value.
pub struct BlockifyResult {
    pub model: TypedModel,
    pub stream_sym: Symbol,
    pub pulse: TDim,
}

/// Find the block-diagonal pattern + rewrite it.  Returns the (possibly
/// unchanged) model and the symbol/pulse to use for pulsification.
pub fn blockify(model: TypedModel, stream_sym: Symbol, pulse: TDim) -> TractResult<BlockifyResult> {
    let sections = find_quadratic_sections(&model, &stream_sym)?;
    let section = match sections.len() {
        0 => return Ok(BlockifyResult { model, stream_sym, pulse }),
        1 => sections.into_iter().next().unwrap(),
        n => bail!(
            "Blockify found {n} independent quadratic sections; multi-section \
             rewriting is not yet implemented.  Refusing to blockify rather \
             than produce a partial rewrite."
        ),
    };
    match rewrite(&model, &stream_sym, &section, &pulse)? {
        Some((new_model, chunk_sym, translated_pulse)) => {
            Ok(BlockifyResult { model: new_model, stream_sym: chunk_sym, pulse: translated_pulse })
        }
        None => Ok(BlockifyResult { model, stream_sym, pulse }),
    }
}

/// Streaming-axis positions on a typed fact.
fn streaming_positions(fact: &TypedFact, stream_sym: &Symbol) -> TVec<usize> {
    fact.shape
        .iter()
        .enumerate()
        .filter(|(_, d)| d.symbols().contains(stream_sym))
        .map(|(i, _)| i)
        .collect()
}

/// A connected subgraph of the typed model where every wire has multi-T-axis
/// shape (≥2 streaming-symbol axes), bracketed by single-T-axis wires.
///
/// Phase 1+2+3 of Blockify recognition produces this structure op-agnostically.
/// Phase 4 (the rewrite) consumes it and dispatches per op-type.
#[derive(Debug)]
struct QuadraticSection {
    /// All nodes whose output wire has multi-T-axis shape.  The rewriter
    /// reads `initiators`/`terminators` directly today; the full set is
    /// kept here because phase 4 body-chain handling (Softmax, Add, etc.)
    /// will need to walk it.
    #[allow(dead_code)]
    section: std::collections::BTreeSet<usize>,
    /// Subset of `section` whose inputs are all outside it (= "rise to quadratic").
    initiators: Vec<usize>,
    /// Nodes outside `section` consuming an in-section wire (= "drop back to linear").
    terminators: Vec<usize>,
    /// Block size extracted from a recognisable mask in the section.
    chunk_size: i64,
}

/// Connected components over the subgraph induced by `nodes` on the model's
/// dataflow.  Two nodes in `nodes` are in the same component iff one's
/// output is consumed by the other.  Returns one `BTreeSet` per component,
/// ordered by smallest node id.
fn connected_components(
    model: &TypedModel,
    nodes: &std::collections::BTreeSet<usize>,
) -> Vec<std::collections::BTreeSet<usize>> {
    use std::collections::BTreeSet;
    let mut parent: HashMap<usize, usize> = nodes.iter().map(|&n| (n, n)).collect();
    fn uf_find(p: &mut HashMap<usize, usize>, x: usize) -> usize {
        let px = p[&x];
        if px == x {
            return x;
        }
        let r = uf_find(p, px);
        p.insert(x, r);
        r
    }
    fn uf_union(p: &mut HashMap<usize, usize>, x: usize, y: usize) {
        let rx = uf_find(p, x);
        let ry = uf_find(p, y);
        if rx != ry {
            p.insert(rx, ry);
        }
    }
    for &nid in nodes {
        for cons in model.outlet_successors(OutletId::new(nid, 0)) {
            if nodes.contains(&cons.node) {
                uf_union(&mut parent, nid, cons.node);
            }
        }
    }
    let mut groups: HashMap<usize, BTreeSet<usize>> = HashMap::default();
    for &nid in nodes {
        let root = uf_find(&mut parent, nid);
        groups.entry(root).or_default().insert(nid);
    }
    let mut out: Vec<BTreeSet<usize>> = groups.into_values().collect();
    out.sort_by_key(|g| *g.iter().next().unwrap_or(&usize::MAX));
    out
}

/// Phase 1+2+3: detect every section of the graph where wires go multi-T-axis.
///
/// The graph may contain several independent quadratic subgraphs (e.g. two
/// attention layers in parallel); each comes back as its own section.  For
/// each candidate section we verify that at least one wire carries
/// `uniform_tdim` or `region_of_interest` (phase 2) and that some wire has a
/// recognisable mask form (phase 3); sections that fail either check are
/// dropped from the result.
fn find_quadratic_sections(
    model: &TypedModel,
    stream_sym: &Symbol,
) -> TractResult<Vec<QuadraticSection>> {
    use std::collections::BTreeSet;

    let is_multi_t_axis = |fact: &TypedFact| {
        fact.shape.iter().filter(|d| d.symbols().contains(stream_sym)).count() >= 2
    };

    // Phase 1a — collect all multi-T-axis nodes.
    let multi_nodes: BTreeSet<usize> = model
        .nodes
        .iter()
        .filter(|n| n.outputs.len() == 1 && is_multi_t_axis(&n.outputs[0].fact))
        .map(|n| n.id)
        .collect();
    if multi_nodes.is_empty() {
        return Ok(vec![]);
    }

    // Phase 1b — connected components over the multi-T-axis subgraph.
    let groups = connected_components(model, &multi_nodes);

    // For each component, run phase 2 + 3.  Drop components that don't have
    // a recognisable mask anchoring them.
    let mut sections: Vec<QuadraticSection> = vec![];
    for section in groups {
        let initiators: Vec<usize> = section
            .iter()
            .copied()
            .filter(|&nid| !model.nodes[nid].inputs.iter().any(|i| section.contains(&i.node)))
            .collect();

        let mut terminators_set: BTreeSet<usize> = BTreeSet::new();
        for &nid in &section {
            for cons in model.outlet_successors(OutletId::new(nid, 0)) {
                if !section.contains(&cons.node) {
                    terminators_set.insert(cons.node);
                }
            }
        }
        let terminators: Vec<usize> = terminators_set.into_iter().collect();

        // Phase 2 — at least one annotated wire.
        let any_annotated = section.iter().any(|&nid| {
            let fact = &model.nodes[nid].outputs[0].fact;
            fact.uniform_tdim.is_some() || fact.region_of_interest.is_some()
        });
        if !any_annotated {
            continue;
        }

        // Phase 3 — recognise a mask form.
        let mut chunk_size: Option<i64> = None;
        for &nid in &section {
            let fact = &model.nodes[nid].outputs[0].fact;
            let Some(uniform) = &fact.uniform_tdim else {
                continue;
            };
            let streaming_axes: TVec<usize> = fact
                .shape
                .iter()
                .enumerate()
                .filter(|(_, d)| d.symbols().contains(stream_sym))
                .map(|(i, _)| i)
                .collect();
            if let Some(k) = decode_block_diag_mask(uniform, &streaming_axes) {
                chunk_size = Some(k);
                break;
            }
        }
        let Some(chunk_size) = chunk_size else {
            continue;
        };

        sections.push(QuadraticSection { section, initiators, terminators, chunk_size });
    }

    Ok(sections)
}

// Pattern is gone — see `rewrite` below, which derives the same per-op
// data from the QuadraticSection on the fly.

/// If `expr` matches `(coord_i / k) == (coord_j / k)` for the same `k`
/// on the two streaming axes, return `k`.
///
/// The recogniser destructures the TDim AST directly so it's robust to
/// `Display` formatting changes and to arbitrary chunk sizes.
fn decode_block_diag_mask(expr: &TDim, streaming_axes: &[usize]) -> Option<i64> {
    if streaming_axes.len() != 2 {
        return None;
    }
    let TDim::Eq(lhs, rhs) = expr else {
        return None;
    };
    let (axis_a, k_a) = decode_coord_div(lhs)?;
    let (axis_b, k_b) = decode_coord_div(rhs)?;
    if k_a != k_b {
        return None;
    }
    // The two coord axes must be exactly the streaming axes (in either order).
    let want: std::collections::BTreeSet<usize> = streaming_axes.iter().copied().collect();
    let got: std::collections::BTreeSet<usize> = [axis_a, axis_b].into_iter().collect();
    if want != got {
        return None;
    }
    Some(k_a as i64)
}

/// Match `Div(Sym(🎯<axis>), k)` and return `(axis, k)`.
fn decode_coord_div(expr: &TDim) -> Option<(usize, u64)> {
    let TDim::Div(num, k) = expr else {
        return None;
    };
    let TDim::Sym(sym) = num.as_ref() else {
        return None;
    };
    let axis = tract_core::ops::logic::sym_to_coord_axis(sym)?;
    Some((axis, *k))
}
/// Rebuild the model in topological order, dispatching each node to a
/// per-role translator: source / outside / initiator / body / terminator
/// (or `dead`, for mask-construction subgraphs that vanish post-rewrite).
///
/// The function reads as "translate the initiators, walk the body, translate
/// the terminators" — except the iteration is the typed model's natural
/// topological order, so each role's nodes get processed in the order their
/// inputs arrive.  Each per-op-type translator is independent: it reads the
/// original node + the shared `mapping` and writes one or more nodes into
/// `out`, recording the new outlet(s) in `mapping`.
///
/// Returns `Ok(None)` if any role hits an op-type the per-op-type
/// dispatchers don't know how to translate; Blockify is a no-op upstream.
fn rewrite(
    model: &TypedModel,
    stream_sym: &Symbol,
    sec: &QuadraticSection,
    pulse: &TDim,
) -> TractResult<Option<(TypedModel, Symbol, TDim)>> {
    let mut out = TypedModel::default();
    out.symbols = model.symbols.clone();
    let chunk_sym = out.symbols.new_with_prefix("S");
    let k = sec.chunk_size;
    let chunk_dim: TDim = chunk_sym.to_dim() * k;
    let mut mapping: HashMap<OutletId, OutletId> = HashMap::default();

    let dead = collect_dead_nodes(model, sec);
    let init_set: std::collections::HashSet<usize> = sec.initiators.iter().copied().collect();
    let term_set: std::collections::HashSet<usize> = sec.terminators.iter().copied().collect();

    for &nid in &model.eval_order()? {
        if dead.contains(&nid) {
            continue;
        }
        let node = &model.nodes[nid];
        let translated = if node.op_is::<tract_core::ops::source::TypedSource>() {
            translate_source(node, stream_sym, &chunk_dim, &mut out, &mut mapping)?
        } else if init_set.contains(&nid) {
            translate_initiator(node, model, stream_sym, &chunk_sym, k, &mut out, &mut mapping)?
        } else if term_set.contains(&nid) {
            translate_terminator(node, model, stream_sym, &chunk_sym, k, &mut out, &mut mapping)?
        } else if sec.section.contains(&nid) {
            translate_body(node, model, &mut mapping)?
        } else {
            translate_outside(node, model, &chunk_sym, &mut out, &mut mapping)?
        };
        if !translated {
            return Ok(None);
        }
    }

    let new_outputs = boundary_merge(model, &chunk_sym, k, &mut out, &mapping)?;
    out.select_output_outlets(&new_outputs)?;

    let translated_pulse =
        pulse.clone().maybe_div(&k.to_dim()).map(|(d, _)| d).map_err(|e| {
            format_err!("Blockify: pulse {pulse} not divisible by chunk size {k}: {e}")
        })?;
    Ok(Some((out, chunk_sym, translated_pulse)))
}

/// Op-agnostic dead-node identification.  A section node whose output has
/// `uniform_tdim` is mask construction (becomes the constant 1 in chunked
/// form, hence dead).  Any node whose only consumers are dead is also dead.
fn collect_dead_nodes(
    model: &TypedModel,
    sec: &QuadraticSection,
) -> std::collections::HashSet<usize> {
    let mut dead: std::collections::HashSet<usize> = sec
        .section
        .iter()
        .copied()
        .filter(|&nid| model.nodes[nid].outputs[0].fact.uniform_tdim.is_some())
        .collect();
    loop {
        let mut changed = false;
        for n in &model.nodes {
            if dead.contains(&n.id) {
                continue;
            }
            let mut consumers = vec![];
            for s in 0..n.outputs.len() {
                for c in model.outlet_successors(OutletId::new(n.id, s)) {
                    consumers.push(c.node);
                }
            }
            if consumers.is_empty() {
                continue; // model output, not dead
            }
            if consumers.iter().all(|c| dead.contains(c)) {
                dead.insert(n.id);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    dead
}

// ── Per-role translators ────────────────────────────────────────────────────
//
// Each returns `Ok(true)` on a successful translation, `Ok(false)` if the
// op-type isn't one the dispatcher knows (Blockify becomes a no-op upstream),
// or `Err` on a hard failure.

fn translate_source(
    node: &TypedNode,
    stream_sym: &Symbol,
    chunk_dim: &TDim,
    out: &mut TypedModel,
    mapping: &mut HashMap<OutletId, OutletId>,
) -> TractResult<bool> {
    let fact = &node.outputs[0].fact;
    let new_shape: TVec<TDim> = fact
        .shape
        .iter()
        .map(|d| d.substitute(stream_sym, chunk_dim))
        .collect::<TractResult<_>>()?;
    let new_fact =
        TypedFact { datum_type: fact.datum_type, shape: new_shape.into(), ..fact.clone() };
    let new_id = out.add_source(node.name.clone(), new_fact)?;
    mapping.insert(OutletId::new(node.id, 0), new_id);
    Ok(true)
}

fn translate_outside(
    node: &TypedNode,
    model: &TypedModel,
    chunk_sym: &Symbol,
    out: &mut TypedModel,
    mapping: &mut HashMap<OutletId, OutletId>,
) -> TractResult<bool> {
    // AxisOp's axis parameters refer to positions in its input.  If the
    // input was chunkified upstream (rank grew by 1, chunk symbol inserted),
    // shift those parameters through the chunk insertion.
    if let Some(axis_op) = node.op_as::<AxisOp>()
        && let Some(shifted) =
            shift_axisop_through_chunk(axis_op, &node.inputs[0], model, chunk_sym, out, mapping)?
    {
        let mapped = mapping[&node.inputs[0]];
        let new_out = out.wire_node(node.name.clone(), shifted, &[mapped])?[0];
        mapping.insert(OutletId::new(node.id, 0), new_out);
        return Ok(true);
    }

    // Default: copy verbatim.
    let inputs: TVec<OutletId> =
        node.inputs
            .iter()
            .map(|i| {
                mapping.get(i).copied().ok_or_else(|| {
                    format_err!("Missing mapping for input {i:?} of node {}", node.name)
                })
            })
            .collect::<TractResult<_>>()?;
    let new_outputs = out.wire_node(node.name.clone(), node.op.clone(), &inputs)?;
    for (slot, &out_id) in new_outputs.iter().enumerate() {
        mapping.insert(OutletId::new(node.id, slot), out_id);
    }
    Ok(true)
}

/// Translate an AxisOp's axis index when its input was chunkified.  Returns
/// `None` when no shift is needed (input rank unchanged), `Some(shifted)`
/// when the chunk axis was inserted into the input upstream and the
/// original axis index needs to slide past it.
fn shift_axisop_through_chunk(
    axis_op: &AxisOp,
    input: &OutletId,
    model: &TypedModel,
    chunk_sym: &Symbol,
    out: &TypedModel,
    mapping: &HashMap<OutletId, OutletId>,
) -> TractResult<Option<AxisOp>> {
    let mapped =
        *mapping.get(input).ok_or_else(|| format_err!("AxisOp input {input:?} not in mapping"))?;
    let mapped_rank = out.outlet_fact(mapped)?.rank();
    let orig_rank = model.outlet_fact(*input)?.rank();
    if mapped_rank <= orig_rank {
        return Ok(None);
    }
    let chunk_pos = out
        .outlet_fact(mapped)?
        .shape
        .iter()
        .position(|d| d == &chunk_sym.to_dim())
        .ok_or_else(|| format_err!("chunked input has no chunk_sym in its shape"))?;
    let shifted = match axis_op {
        AxisOp::Rm(a) => AxisOp::Rm(chunked_axis_index(*a, chunk_pos)),
        AxisOp::Add(a) => AxisOp::Add(chunked_axis_index(*a, chunk_pos)),
        AxisOp::Move(from, to) => {
            AxisOp::Move(chunked_axis_index(*from, chunk_pos), chunked_axis_index(*to, chunk_pos))
        }
        AxisOp::Reshape(at, from, to) => {
            // POC: just shift the `at` index.  Splitting the chunk axis
            // itself is out of scope.
            AxisOp::Reshape(chunked_axis_index(*at, chunk_pos), from.clone(), to.clone())
        }
    };
    Ok(Some(shifted))
}

fn translate_initiator(
    node: &TypedNode,
    model: &TypedModel,
    stream_sym: &Symbol,
    chunk_sym: &Symbol,
    k: i64,
    out: &mut TypedModel,
    mapping: &mut HashMap<OutletId, OutletId>,
) -> TractResult<bool> {
    if node.outputs[0].fact.uniform_tdim.is_some() {
        // Mask-construction initiator (Eq, etc.): handled as dead earlier.
        // Defensive no-op in case this lands here.
        return Ok(true);
    }
    if let Some(es) = node.op_as::<EinSum>() {
        translate_initiator_einsum(node, es, model, stream_sym, chunk_sym, k, out, mapping)?;
        return Ok(true);
    }
    Ok(false)
}

fn translate_initiator_einsum(
    node: &TypedNode,
    op: &EinSum,
    model: &TypedModel,
    stream_sym: &Symbol,
    chunk_sym: &Symbol,
    k: i64,
    out: &mut TypedModel,
    mapping: &mut HashMap<OutletId, OutletId>,
) -> TractResult<()> {
    let mut in_axes: TVec<usize> = tvec!();
    for &input in &node.inputs {
        let positions = streaming_positions(model.outlet_fact(input)?, stream_sym);
        if positions.len() != 1 {
            bail!(
                "Blockify: initiator EinSum input must have exactly one streaming axis (got {})",
                positions.len()
            );
        }
        in_axes.push(positions[0]);
    }
    let out_axes = streaming_positions(&node.outputs[0].fact, stream_sym);
    if out_axes.len() != 2 || out_axes[1] != out_axes[0] + 1 {
        bail!("Blockify: initiator EinSum output must have two contiguous streaming axes");
    }

    let mut chunked_inputs: TVec<OutletId> = tvec!();
    for (ix, &input) in node.inputs.iter().enumerate() {
        let mapped = *mapping
            .get(&input)
            .ok_or_else(|| format_err!("initiator input {input:?} not in mapping"))?;
        let mapped_fact = out.outlet_fact(mapped)?.clone();
        let stream_axis =
            mapped_fact.shape.iter().position(|d| d.symbols().contains(chunk_sym)).ok_or_else(
                || format_err!("initiator input {ix} lost streaming axis after substitute"),
            )?;
        let from = tvec!(mapped_fact.shape[stream_axis].clone());
        let to = tvec!(chunk_sym.to_dim(), k.to_dim());
        let reshape = AxisOp::Reshape(stream_axis, from, to);
        let chunked =
            out.wire_node(format!("{}.blockify_split.{ix}", node.name), reshape, &[mapped])?[0];
        chunked_inputs.push(chunked);
    }

    let in_starts: Vec<Option<usize>> = in_axes.iter().map(|&p| Some(p)).collect();
    let new_einsum = chunkify_einsum(op, &in_starts, Some(out_axes[0]))?;
    let new_out = out.wire_node(node.name.clone(), new_einsum, &chunked_inputs)?[0];
    mapping.insert(OutletId::new(node.id, 0), new_out);
    Ok(())
}

fn translate_body(
    node: &TypedNode,
    model: &TypedModel,
    mapping: &mut HashMap<OutletId, OutletId>,
) -> TractResult<bool> {
    // Mul-by-mask: the body Mul where one input is uniform_tdim (the mask).
    // The chunked graph multiplies by 1, so the output IS the chunked compute
    // input.  We don't emit a node — just record the alias in `mapping`.
    if let Some(bin) = node.op_as::<TypedBinOp>()
        && bin.0.is::<Mul>()
    {
        let compute_input = node
            .inputs
            .iter()
            .copied()
            .find(|i| model.outlet_fact(*i).map(|f| f.uniform_tdim.is_none()).unwrap_or(true));
        if let Some(compute_input) = compute_input
            && let Some(&chunked) = mapping.get(&compute_input)
        {
            mapping.insert(OutletId::new(node.id, 0), chunked);
            return Ok(true);
        }
    }
    Ok(false)
}

fn translate_terminator(
    node: &TypedNode,
    model: &TypedModel,
    stream_sym: &Symbol,
    chunk_sym: &Symbol,
    k: i64,
    out: &mut TypedModel,
    mapping: &mut HashMap<OutletId, OutletId>,
) -> TractResult<bool> {
    if let Some(reduce_op) = node.op_as::<Reduce>() {
        translate_terminator_reduce(node, reduce_op, model, stream_sym, out, mapping)?;
        return Ok(true);
    }
    if let Some(es) = node.op_as::<EinSum>() {
        translate_terminator_einsum(node, es, model, stream_sym, chunk_sym, k, out, mapping)?;
        return Ok(true);
    }
    Ok(false)
}

fn translate_terminator_reduce(
    node: &TypedNode,
    op: &Reduce,
    model: &TypedModel,
    stream_sym: &Symbol,
    out: &mut TypedModel,
    mapping: &mut HashMap<OutletId, OutletId>,
) -> TractResult<()> {
    if op.reducer != Reducer::Sum || op.axes.len() != 1 {
        bail!("Blockify: only Reduce<Sum> on a single axis is supported as terminator");
    }
    // Chunk insertion position is the first streaming axis of the input
    // (the multi-T-axis wire from the body Mul).  Derived from the original
    // input fact, applied through `chunked_axis_index` to the reduce axis.
    let in_streaming = streaming_positions(model.outlet_fact(node.inputs[0])?, stream_sym);
    let chunk_pos = *in_streaming
        .first()
        .ok_or_else(|| format_err!("Reduce terminator input has no streaming axis"))?;
    let mapped_input = *mapping
        .get(&node.inputs[0])
        .ok_or_else(|| format_err!("Reduce terminator input not in mapping"))?;
    let new_axis = chunked_axis_index(op.axes[0], chunk_pos);
    let new_reduce = Reduce { axes: tvec!(new_axis), reducer: op.reducer };
    let new_out = out.wire_node(node.name.clone(), new_reduce, &[mapped_input])?[0];
    mapping.insert(OutletId::new(node.id, 0), new_out);
    Ok(())
}

fn translate_terminator_einsum(
    node: &TypedNode,
    op: &EinSum,
    model: &TypedModel,
    stream_sym: &Symbol,
    chunk_sym: &Symbol,
    k: i64,
    out: &mut TypedModel,
    mapping: &mut HashMap<OutletId, OutletId>,
) -> TractResult<()> {
    let mut chunked_inputs: TVec<OutletId> = tvec!();
    let mut input_starts: Vec<Option<usize>> = vec![];
    for (slot, &input) in node.inputs.iter().enumerate() {
        let positions = streaming_positions(model.outlet_fact(input)?, stream_sym);
        let mapped = *mapping
            .get(&input)
            .ok_or_else(|| format_err!("EinSum terminator input {input:?} not in mapping"))?;
        match positions.len() {
            0 => {
                chunked_inputs.push(mapped);
                input_starts.push(None);
            }
            1 => {
                // Single-T-axis auxiliary: chunkify via a Reshape on the streaming axis.
                let mapped_fact = out.outlet_fact(mapped)?.clone();
                let stream_axis = mapped_fact
                    .shape
                    .iter()
                    .position(|d| d.symbols().contains(chunk_sym))
                    .ok_or_else(|| {
                        format_err!("Single-T-axis input lost streaming axis after substitute")
                    })?;
                let from = tvec!(mapped_fact.shape[stream_axis].clone());
                let to = tvec!(chunk_sym.to_dim(), k.to_dim());
                let reshape = AxisOp::Reshape(stream_axis, from, to);
                let chunked = out.wire_node(
                    format!("{}.blockify_split.in{slot}", node.name),
                    reshape,
                    &[mapped],
                )?[0];
                chunked_inputs.push(chunked);
                input_starts.push(Some(positions[0]));
            }
            2 => {
                // Multi-T-axis input: already in chunked form via mapping
                // (the body Mul-by-mask was shunted to alias the chunked
                // initiator output).
                chunked_inputs.push(mapped);
                input_starts.push(Some(positions[0]));
            }
            _ => bail!(
                "Blockify: EinSum terminator input with {} streaming axes is unsupported",
                positions.len()
            ),
        }
    }

    let out_streaming = streaming_positions(&node.outputs[0].fact, stream_sym);
    let new_einsum = chunkify_einsum(op, &input_starts, out_streaming.first().copied())?;
    let new_out = out.wire_node(node.name.clone(), new_einsum, &chunked_inputs)?[0];
    mapping.insert(OutletId::new(node.id, 0), new_out);
    Ok(())
}

/// Boundary merge: at each model output, if the wire still has the chunk
/// axis exposed followed by the within-chunk axis of size k, flatten them
/// back into a single k·S axis at the same position — restores the
/// original streaming-axis layout without changing the model interface.
fn boundary_merge(
    model: &TypedModel,
    chunk_sym: &Symbol,
    k: i64,
    out: &mut TypedModel,
    mapping: &HashMap<OutletId, OutletId>,
) -> TractResult<TVec<OutletId>> {
    let mut new_outputs: TVec<OutletId> = tvec!();
    for &orig in model.output_outlets()? {
        let mut wire = mapping
            .get(&orig)
            .copied()
            .ok_or_else(|| format_err!("Missing mapping for output {orig:?}"))?;
        let fact = out.outlet_fact(wire)?.clone();
        let chunk_pos = fact.shape.iter().position(|d| d == &chunk_sym.to_dim());
        if let Some(pos) = chunk_pos
            && pos + 1 < fact.shape.len()
            && fact.shape[pos + 1] == k.to_dim()
        {
            let from = tvec!(chunk_sym.to_dim(), k.to_dim());
            let to = tvec!(chunk_sym.to_dim() * k);
            let reshape = AxisOp::Reshape(pos, from, to);
            wire = out.wire_node(
                format!("{}.blockify_merge", model.nodes[orig.node].name),
                reshape,
                &[wire],
            )?[0];
        }
        new_outputs.push(wire);
    }
    Ok(new_outputs)
}

fn chunked_axis_index(orig_axis: usize, chunk_pos: usize) -> usize {
    // The chunk batch axis is inserted at `chunk_pos` (the position of the
    // first streaming output axis).  Every original axis at or after that
    // position shifts right by one.  Streaming axes themselves are no
    // exception — they shift by one too, because the chunk axis lives where
    // they used to start.
    if orig_axis < chunk_pos { orig_axis } else { orig_axis + 1 }
}

/// Insert the chunk-axis char at the streaming-axis position on each input
/// and output.  `None` for an input/output skips the insertion (no streaming
/// axis there, so no chunk axis on that side).  Within-chunk versions of
/// formerly-streaming axes keep their original chars and shift right by 1.
fn chunkify_einsum(
    op: &EinSum,
    input_streaming_starts: &[Option<usize>],
    output_streaming_start: Option<usize>,
) -> TractResult<EinSum> {
    let (inputs, outputs) = op.axes.to_strs();
    let new_repr = pick_free_axis_repr(&op.axes);
    let insert_at = |s: &String, pos: Option<usize>| -> String {
        let Some(p) = pos else {
            return s.clone();
        };
        let mut chars: Vec<char> = s.chars().collect();
        chars.insert(p, new_repr);
        chars.into_iter().collect()
    };
    let new_inputs: Vec<String> = inputs
        .iter()
        .zip(input_streaming_starts.iter())
        .map(|(s, &pos)| insert_at(s, pos))
        .collect();
    let new_outputs: Vec<String> = outputs
        .iter()
        .enumerate()
        .map(|(i, s)| if i == 0 { insert_at(s, output_streaming_start) } else { s.clone() })
        .collect();
    let new_mapping = AxesMapping::from_strs(&new_inputs, &new_outputs)?;
    Ok(EinSum { axes: new_mapping, operating_dt: op.operating_dt, q_params: op.q_params.clone() })
}

fn pick_free_axis_repr(axes: &AxesMapping) -> char {
    let used: std::collections::HashSet<char> = axes.iter_all_axes().map(|a| a.repr).collect();
    for c in 'a'..='z' {
        if !used.contains(&c) {
            return c;
        }
    }
    'Z'
}

#[cfg(test)]
mod tests {
    use super::*;

    fn coord(scope: &SymbolScope, axis: usize) -> TDim {
        TDim::Sym(scope.sym(&format!("🎯{axis}")))
    }

    fn make_block_diag(scope: &SymbolScope, i: usize, j: usize, k: u64) -> TDim {
        TDim::Eq(
            Box::new(TDim::Div(Box::new(coord(scope, i)), k)),
            Box::new(TDim::Div(Box::new(coord(scope, j)), k)),
        )
    }

    #[test]
    fn decode_block_diag_mask_recognises_canonical_form() {
        let scope = SymbolScope::default();
        let expr = make_block_diag(&scope, 0, 1, 2);
        assert_eq!(decode_block_diag_mask(&expr, &[0, 1]), Some(2));
    }

    #[test]
    fn decode_block_diag_mask_recognises_arbitrary_chunk_size() {
        let scope = SymbolScope::default();
        let expr = make_block_diag(&scope, 0, 1, 137);
        assert_eq!(decode_block_diag_mask(&expr, &[0, 1]), Some(137));
    }

    #[test]
    fn decode_block_diag_mask_recognises_swapped_axes() {
        let scope = SymbolScope::default();
        // (🎯1)/2 == (🎯0)/2 — same relation, syms in opposite order.
        let expr = make_block_diag(&scope, 1, 0, 2);
        assert_eq!(decode_block_diag_mask(&expr, &[0, 1]), Some(2));
    }

    #[test]
    fn decode_block_diag_mask_rejects_mismatched_chunk_sizes() {
        let scope = SymbolScope::default();
        let expr = TDim::Eq(
            Box::new(TDim::Div(Box::new(coord(&scope, 0)), 2)),
            Box::new(TDim::Div(Box::new(coord(&scope, 1)), 3)),
        );
        assert_eq!(decode_block_diag_mask(&expr, &[0, 1]), None);
    }

    #[test]
    fn decode_block_diag_mask_rejects_non_streaming_axis() {
        let scope = SymbolScope::default();
        // Mask references coord 2 instead of one of the streaming axes [0, 1].
        let expr = make_block_diag(&scope, 0, 2, 2);
        assert_eq!(decode_block_diag_mask(&expr, &[0, 1]), None);
    }

    #[test]
    fn decode_block_diag_mask_rejects_non_eq_root() {
        let scope = SymbolScope::default();
        let expr = TDim::Ge(
            Box::new(TDim::Div(Box::new(coord(&scope, 0)), 2)),
            Box::new(TDim::Div(Box::new(coord(&scope, 1)), 2)),
        );
        assert_eq!(decode_block_diag_mask(&expr, &[0, 1]), None);
    }

    #[test]
    fn decode_block_diag_mask_rejects_offset_in_numerator() {
        let scope = SymbolScope::default();
        // ((🎯0 + 1) / 2 == (🎯1 + 1) / 2) — algebraically the same chunk-shift
        // pattern, but our recogniser intentionally requires the canonical
        // post-declutter form.  If declutter starts producing this variant we
        // will need to extend the recogniser.
        let expr = TDim::Eq(
            Box::new(TDim::Div(Box::new(TDim::Add(vec![coord(&scope, 0), TDim::Val(1)])), 2)),
            Box::new(TDim::Div(Box::new(TDim::Add(vec![coord(&scope, 1), TDim::Val(1)])), 2)),
        );
        assert_eq!(decode_block_diag_mask(&expr, &[0, 1]), None);
    }

    fn einsum_for(inputs: &[&str], output: &str) -> EinSum {
        EinSum {
            axes: AxesMapping::from_strs(inputs, &[output]).unwrap(),
            operating_dt: f32::datum_type(),
            q_params: None,
        }
    }

    fn axes_to_strings(op: &EinSum) -> (Vec<String>, Vec<String>) {
        let (ins, outs) = op.axes.to_strs();
        (ins.into_iter().collect(), outs.into_iter().collect())
    }

    fn ck(op: &EinSum, ins: &[usize], out: usize) -> EinSum {
        let in_starts: Vec<Option<usize>> = ins.iter().map(|&p| Some(p)).collect();
        chunkify_einsum(op, &in_starts, Some(out)).unwrap()
    }

    #[test]
    fn chunkify_einsum_handles_streaming_at_position_zero() {
        let op = einsum_for(&["id", "jd"], "ij");
        let chunked = ck(&op, &[0, 0], 0);
        let (ins, outs) = axes_to_strings(&chunked);
        let chunk_char = pick_free_axis_repr(&op.axes);
        assert_eq!(ins[0], format!("{chunk_char}id"));
        assert_eq!(ins[1], format!("{chunk_char}jd"));
        assert_eq!(outs[0], format!("{chunk_char}ij"));
    }

    #[test]
    fn chunkify_einsum_handles_streaming_at_inner_position() {
        let op = einsum_for(&["bid", "bjd"], "bij");
        let chunked = ck(&op, &[1, 1], 1);
        let (ins, outs) = axes_to_strings(&chunked);
        let chunk_char = pick_free_axis_repr(&op.axes);
        assert_eq!(ins[0], format!("b{chunk_char}id"));
        assert_eq!(ins[1], format!("b{chunk_char}jd"));
        assert_eq!(outs[0], format!("b{chunk_char}ij"));
    }

    #[test]
    fn chunkify_einsum_handles_mixed_input_positions() {
        let op = einsum_for(&["id", "bjd"], "bij");
        let chunked = ck(&op, &[0, 1], 1);
        let (ins, outs) = axes_to_strings(&chunked);
        let chunk_char = pick_free_axis_repr(&op.axes);
        assert_eq!(ins[0], format!("{chunk_char}id"));
        assert_eq!(ins[1], format!("b{chunk_char}jd"));
        assert_eq!(outs[0], format!("b{chunk_char}ij"));
    }

    #[test]
    fn chunkify_einsum_for_terminator_with_two_streaming_input() {
        // ex02 terminator: "ij,jd->id".  Input 0 (masked) has streaming at
        // positions 0 and 1 — chunk char goes before position 0.  Input 1
        // (c) has streaming at position 0.  Output has streaming at 0.
        let op = einsum_for(&["ij", "jd"], "id");
        let chunked = ck(&op, &[0, 0], 0);
        let (ins, outs) = axes_to_strings(&chunked);
        let chunk_char = pick_free_axis_repr(&op.axes);
        assert_eq!(ins[0], format!("{chunk_char}ij"));
        assert_eq!(ins[1], format!("{chunk_char}jd"));
        assert_eq!(outs[0], format!("{chunk_char}id"));
    }

    #[test]
    fn chunked_axis_index_zero_chunk_position() {
        // Chunk at output pos 0; everything shifts right by 1.
        assert_eq!(chunked_axis_index(0, 0), 1);
        assert_eq!(chunked_axis_index(1, 0), 2);
    }

    #[test]
    fn chunked_axis_index_inner_chunk_position() {
        // Chunk at output pos 1; axis 0 stays, axes 1+ shift right.
        assert_eq!(chunked_axis_index(0, 1), 0);
        assert_eq!(chunked_axis_index(1, 1), 2);
        assert_eq!(chunked_axis_index(2, 1), 3);
    }

    /// Build a tiny model with two parallel chains of identity ops, all
    /// claiming multi-T-axis shape via hand-crafted facts, and check that
    /// the connected-components walker splits them.
    #[test]
    fn connected_components_splits_independent_subgraphs() {
        use std::collections::BTreeSet;
        // Build the model topologically by hand: two parallel pairs of
        // sources, each consumed by an identity-ish op (we use AxisOp::Add
        // to add a unit axis — its output shape doesn't actually matter for
        // this test; we only edit the fact afterwards to make the walker
        // see multi-T-axis on selected nodes).
        let mut model = TypedModel::default();
        let t = model.symbols.sym("T");

        let a1 = model.add_source("a1", f32::fact(dims![t.clone(), 4_usize].as_ref())).unwrap();
        let b1 =
            model.wire_node("b1", tract_core::ops::change_axes::AxisOp::Add(0), &[a1]).unwrap()[0];
        let c1 =
            model.wire_node("c1", tract_core::ops::change_axes::AxisOp::Add(0), &[b1]).unwrap()[0];

        let a2 = model.add_source("a2", f32::fact(dims![t.clone(), 4_usize].as_ref())).unwrap();
        let b2 =
            model.wire_node("b2", tract_core::ops::change_axes::AxisOp::Add(0), &[a2]).unwrap()[0];
        let c2 =
            model.wire_node("c2", tract_core::ops::change_axes::AxisOp::Add(0), &[b2]).unwrap()[0];

        model.select_output_outlets(&[c1, c2]).unwrap();

        // Pretend nodes b1, c1, b2, c2 are multi-T-axis (the function we're
        // testing only inspects connectivity; it doesn't look at facts).
        let multi: BTreeSet<usize> = [b1.node, c1.node, b2.node, c2.node].into_iter().collect();
        let groups = connected_components(&model, &multi);
        assert_eq!(groups.len(), 2, "expected two independent components: {groups:?}");
        // Each component must contain exactly its two nodes.
        let g0: BTreeSet<usize> = [b1.node, c1.node].into_iter().collect();
        let g1: BTreeSet<usize> = [b2.node, c2.node].into_iter().collect();
        assert!(groups.iter().any(|g| *g == g0), "expected component {g0:?} in {groups:?}");
        assert!(groups.iter().any(|g| *g == g1), "expected component {g1:?} in {groups:?}");
    }
}
