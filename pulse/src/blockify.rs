//! Blockify — typed-model rewrite that factors block-diagonal structure
//! into the graph topology, so the result has a single streaming axis
//! everywhere and pulsifies under v1's existing machinery.
//!
//! Recogniser scope: a single block-diagonal pattern.
//!
//!   EinSum([a, b]) producing scores[T, T]
//!   → Mul(scores, mask) where mask has uniform_tdim `(coord_a/k == coord_b/k)`
//!   → Reduce<Sum> on a streaming axis  OR  contracting EinSum (ex02)
//!
//! Implementation layout: detect quadratic sections globally, substitute
//! the streaming symbol T → k·S via core's `substitute_symbols`, then
//! build one TypedModelPatch per section that does the chunkification
//! locally and shunts the boundary outlet to the chunked-then-merged
//! output.  Sections are independent, so patches apply in sequence.

use crate::internal::*;
use std::collections::{BTreeSet, HashMap, HashSet};
use tract_core::axes::AxesMapping;
use tract_core::model::TypedModelPatch;
use tract_core::ops::binary::TypedBinOp;
use tract_core::ops::change_axes::AxisOp;
use tract_core::ops::einsum::EinSum;
use tract_core::ops::math::Mul;
use tract_core::ops::nn::{Reduce, Reducer};
use tract_core::transform::ModelTransform;

/// Configuration for the Blockify ModelTransform.
#[derive(Debug, Default, Clone, serde::Deserialize)]
pub struct BlockifyConfig {
    /// Streaming symbol the model's quadratic sections are quadratic in.
    /// Defaults to "S" (matches the convention used by the pulse transform).
    pub symbol: Option<String>,
}

/// Property key holding the symbol introduced by a Blockify rewrite — the
/// new (chunk-counting) streaming symbol that downstream consumers (e.g.
/// the pulse transform) should use.  Stored as a 1-element string tensor.
pub const BLOCKIFY_CHUNK_SYMBOL: &str = "blockify.chunk_symbol";

/// Property key holding the chunk size `k`.  Pulse values originally
/// expressed in token-units must be divided by this to convert to
/// chunk-units after Blockify runs.  Stored as a scalar i64 tensor.
pub const BLOCKIFY_CHUNK_SIZE: &str = "blockify.chunk_size";

/// Property key holding the original (pre-substitution) streaming symbol
/// name.  Mostly informational.  Stored as a 1-element string tensor.
pub const BLOCKIFY_ORIGINAL_SYMBOL: &str = "blockify.original_symbol";

#[derive(Debug)]
pub struct BlockifyTransform(pub BlockifyConfig);

impl ModelTransform for BlockifyTransform {
    fn name(&self) -> std::borrow::Cow<'static, str> {
        "blockify".into()
    }

    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        let symbol_name = self.0.symbol.as_deref().unwrap_or("S");
        let stream_sym = model.symbols.sym(symbol_name);
        let sections = find_quadratic_sections(model, &stream_sym)?;
        if sections.is_empty() {
            return Ok(());
        }
        let k = sections[0].chunk_size;
        if !sections.iter().all(|s| s.chunk_size == k) {
            bail!(
                "Blockify found multiple quadratic sections with mismatched chunk \
                 sizes; a single global symbol substitution cannot cover them.  \
                 Refusing to blockify rather than produce a partial rewrite."
            );
        }

        // Phase 1: introduce S, substitute T → k·S globally via core.
        let chunk_sym = model.symbols.new_with_prefix("S");
        let subs: HashMap<Symbol, TDim> =
            HashMap::from([(stream_sym.clone(), chunk_sym.to_dim() * k)]);
        let mut new_model = model.substitute_symbols(&subs)?;

        // Phase 2: one TypedModelPatch per section.
        for sec in &sections {
            let Some(patch) = build_section_patch(&new_model, sec, &chunk_sym, k)? else {
                continue;
            };
            patch.apply(&mut new_model)?;
        }

        // Ancillary outputs — describe the substitution for downstream
        // consumers (e.g. the pulse transform that needs to translate its
        // pulse value from token-units to chunk-units).
        new_model.properties.insert(
            BLOCKIFY_CHUNK_SYMBOL.to_string(),
            tensor1(&[format!("{chunk_sym}")]).into_arc_tensor(),
        );
        new_model.properties.insert(BLOCKIFY_CHUNK_SIZE.to_string(), tensor0(k).into_arc_tensor());
        new_model.properties.insert(
            BLOCKIFY_ORIGINAL_SYMBOL.to_string(),
            tensor1(&[symbol_name.to_string()]).into_arc_tensor(),
        );

        *model = new_model;
        Ok(())
    }
}

/// Read back the `(chunk_symbol, chunk_size)` ancillary outputs that
/// `BlockifyTransform` writes to model properties.  Returns `None` if the
/// model wasn't blockified (or those properties aren't present).
pub fn blockify_output(model: &TypedModel) -> Option<(Symbol, i64)> {
    let k = model.properties.get(BLOCKIFY_CHUNK_SIZE)?.cast_to_scalar::<i64>().ok()?;
    let name_tensor = model.properties.get(BLOCKIFY_CHUNK_SYMBOL)?;
    let view = name_tensor.to_plain_array_view::<String>().ok()?;
    let name = view.iter().next()?;
    Some((model.symbols.sym(name), k))
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
    section: BTreeSet<usize>,
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
fn connected_components(model: &TypedModel, nodes: &BTreeSet<usize>) -> Vec<BTreeSet<usize>> {
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
    let want: BTreeSet<usize> = streaming_axes.iter().copied().collect();
    let got: BTreeSet<usize> = [axis_a, axis_b].into_iter().collect();
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

/// Build a TypedModelPatch that chunkifies one quadratic section.
///
/// Reads as "iterate initiators, walk the body, iterate terminators" — each
/// role iterates op-agnostically over `sec.initiators` / section nodes /
/// `sec.terminators` and dispatches to per-op-type sub-functions that wire
/// the chunked equivalent into the patch.  Returns `Ok(None)` if any role
/// hits an op-type the dispatchers don't handle (the section is left alone;
/// downstream pulsification will fail with a clear error).
fn build_section_patch(
    model: &TypedModel,
    sec: &QuadraticSection,
    chunk_sym: &Symbol,
    k: i64,
) -> TractResult<Option<TypedModelPatch>> {
    let mut patch = TypedModelPatch::default();
    // Map from original outlet to its chunked equivalent inside the patch.
    let mut chunked: HashMap<OutletId, OutletId> = HashMap::default();
    // Boundary outlets to redirect via `shunt_outside` after wiring the
    // merge reshape: (original outlet, chunked-form outlet inside patch).
    let mut shunts: Vec<(OutletId, OutletId)> = vec![];

    // ── 1. Initiators ────────────────────────────────────────────────────
    for &nid in &sec.initiators {
        let node = &model.nodes[nid];
        // Mask-construction initiators (uniform_tdim outputs) are dead
        // post-rewrite; they're handled by the obliterate pass at the end.
        if node.outputs[0].fact.uniform_tdim.is_some() {
            continue;
        }
        rule_if_some!(out = wire_initiator(&mut patch, model, node, chunk_sym, k)?);
        chunked.insert(OutletId::new(nid, 0), out);
    }
    rule_if!(!chunked.is_empty());

    // ── 2. Body ──────────────────────────────────────────────────────────
    // Walk the section in topological order, skipping initiators (already
    // wired), terminators (out-of-section by definition), and uniform_tdim
    // wires (dead).
    for &nid in &model.eval_order()? {
        if !sec.section.contains(&nid) {
            continue;
        }
        if sec.initiators.contains(&nid) {
            continue;
        }
        let node = &model.nodes[nid];
        if node.outputs[0].fact.uniform_tdim.is_some() {
            continue;
        }
        rule_if_some!(out = wire_body(model, node, &chunked)?);
        chunked.insert(OutletId::new(nid, 0), out);
    }

    // ── 3. Terminators ───────────────────────────────────────────────────
    for &nid in &sec.terminators {
        let node = &model.nodes[nid];
        rule_if_some!(
            (boundary, chunked_form) =
                wire_terminator(&mut patch, model, node, &chunked, chunk_sym, k)?
        );
        shunts.push((boundary, chunked_form));
    }

    // ── 4. Boundary merges + shunts ──────────────────────────────────────
    for (boundary, chunked_form) in shunts {
        let merged = wire_merge_reshape(
            &mut patch,
            &model.nodes[boundary.node].name,
            chunked_form,
            chunk_sym,
            k,
        )?;
        patch.shunt_outside(model, boundary, merged)?;
    }

    // ── 5. Obliterate dead nodes ─────────────────────────────────────────
    // A node is dead iff its output (a) has uniform_tdim and is in the
    // section (= mask construction), or (b) all its consumers are dead
    // (transitively).  Plus the initiators / body / terminators / post-
    // terminator boundary nodes we just rewired.
    let dead = collect_dead_nodes(model, sec, &patch.shunts);
    for nid in dead {
        patch.obliterate(nid)?;
    }

    Ok(Some(patch))
}

// ── Per-role dispatchers ────────────────────────────────────────────────
//
// Each `wire_*` helper takes a section node + the patch-in-progress and
// dispatches to a per-op-type implementation.  Returning `Ok(None)` means
// "I don't know how to handle this op-type"; the caller bubbles that up
// to the section-patch level which then refuses the rewrite.

fn wire_initiator(
    patch: &mut TypedModelPatch,
    model: &TypedModel,
    node: &TypedNode,
    chunk_sym: &Symbol,
    k: i64,
) -> TractResult<Option<OutletId>> {
    if let Some(op) = node.op_as::<EinSum>() {
        return Ok(Some(wire_initiator_einsum(patch, model, node, op, chunk_sym, k)?));
    }
    Ok(None)
}

fn wire_body(
    model: &TypedModel,
    node: &TypedNode,
    chunked: &HashMap<OutletId, OutletId>,
) -> TractResult<Option<OutletId>> {
    if let Some(bin) = node.op_as::<TypedBinOp>()
        && bin.0.is::<Mul>()
    {
        return Ok(wire_body_mul_by_mask(model, node, chunked));
    }
    Ok(None)
}

fn wire_terminator(
    patch: &mut TypedModelPatch,
    model: &TypedModel,
    node: &TypedNode,
    chunked: &HashMap<OutletId, OutletId>,
    chunk_sym: &Symbol,
    k: i64,
) -> TractResult<Option<(OutletId, OutletId)>> {
    if let Some(op) = node.op_as::<Reduce>() {
        return wire_terminator_reduce(patch, model, node, op, chunked);
    }
    if let Some(op) = node.op_as::<EinSum>() {
        return wire_terminator_einsum(patch, model, node, op, chunked, chunk_sym, k);
    }
    Ok(None)
}

// ── Per-op-type implementations ─────────────────────────────────────────

/// Initiator EinSum: tap each input from the model, wire a split reshape
/// for it, then wire the chunked EinSum.  Returns the chunked output.
fn wire_initiator_einsum(
    patch: &mut TypedModelPatch,
    model: &TypedModel,
    node: &TypedNode,
    op: &EinSum,
    chunk_sym: &Symbol,
    k: i64,
) -> TractResult<OutletId> {
    let out_streaming_axes = streaming_positions(&node.outputs[0].fact, chunk_sym);
    ensure!(
        out_streaming_axes.len() == 2 && out_streaming_axes[1] == out_streaming_axes[0] + 1,
        "Initiator EinSum output must have two contiguous streaming axes"
    );
    let mut in_streaming_axes: TVec<usize> = tvec!();
    for &input in &node.inputs {
        let positions = streaming_positions(model.outlet_fact(input)?, chunk_sym);
        ensure!(
            positions.len() == 1,
            "Initiator EinSum input must have exactly one streaming axis"
        );
        in_streaming_axes.push(positions[0]);
    }

    let mut chunked_inputs: TVec<OutletId> = tvec!();
    for (ix, (&input, &stream_axis)) in node.inputs.iter().zip(in_streaming_axes.iter()).enumerate()
    {
        let tapped = patch.tap_model(model, input)?;
        let in_fact = patch.outlet_fact(tapped)?.clone();
        let from = tvec!(in_fact.shape[stream_axis].clone());
        let to = tvec!(chunk_sym.to_dim(), k.to_dim());
        let reshape = AxisOp::Reshape(stream_axis, from, to);
        let chunked =
            patch.wire_node(format!("{}.blockify_split.{ix}", node.name), reshape, &[tapped])?[0];
        chunked_inputs.push(chunked);
    }

    let in_starts: Vec<Option<usize>> = in_streaming_axes.iter().map(|&p| Some(p)).collect();
    let chunked_op = chunkify_einsum(op, &in_starts, Some(out_streaming_axes[0]))?;
    Ok(patch.wire_node(format!("{}.blockified", node.name), chunked_op, &chunked_inputs)?[0])
}

/// Body Mul-by-mask: aliases the chunked compute input as the chunked
/// output (mask is all-1 in chunked form, Mul disappears).  No node is
/// added to the patch; the alias lives in the `chunked` map.
fn wire_body_mul_by_mask(
    model: &TypedModel,
    node: &TypedNode,
    chunked: &HashMap<OutletId, OutletId>,
) -> Option<OutletId> {
    let compute_input = node
        .inputs
        .iter()
        .copied()
        .find(|i| model.outlet_fact(*i).map(|f| f.uniform_tdim.is_none()).unwrap_or(true))?;
    chunked.get(&compute_input).copied()
}

/// Reduce<Sum> terminator: wires a chunked Reduce on the within-chunk
/// version of the original reduce axis.  If a downstream `RmAxis` removes
/// the now-size-1 reduced slot, wire its chunked counterpart inside the
/// patch and use its output as the boundary.
fn wire_terminator_reduce(
    patch: &mut TypedModelPatch,
    model: &TypedModel,
    node: &TypedNode,
    op: &Reduce,
    chunked: &HashMap<OutletId, OutletId>,
) -> TractResult<Option<(OutletId, OutletId)>> {
    rule_if!(op.reducer == Reducer::Sum && op.axes.len() == 1);
    rule_if_some!(chunked_input = chunked.get(&node.inputs[0]).copied());
    // Chunk insertion position: the first streaming axis of the input fact.
    let in_fact = model.outlet_fact(node.inputs[0])?;
    rule_if_some!(stream_sym = first_streaming_symbol(in_fact));
    let in_streaming = streaming_positions(in_fact, &stream_sym);
    rule_if!(!in_streaming.is_empty());
    let chunk_pos = in_streaming[0];
    let new_axis = chunked_axis_index(op.axes[0], chunk_pos);
    let new_reduce = Reduce { axes: tvec!(new_axis), reducer: op.reducer };
    let chunked_term =
        patch.wire_node(format!("{}.blockified", node.name), new_reduce, &[chunked_input])?[0];

    // If the immediate consumer is `AxisOp::Rm` on the (former) reduce
    // axis, wire its chunked counterpart and use its output as the
    // boundary.  Otherwise the Reduce's own output is the boundary.
    let term_consumers = model.outlet_successors(OutletId::new(node.id, 0));
    if term_consumers.len() == 1 {
        let consumer = &model.nodes[term_consumers[0].node];
        if let Some(AxisOp::Rm(axis)) = consumer.op_as::<AxisOp>()
            && *axis == op.axes[0]
        {
            let new_axis = chunked_axis_index(op.axes[0], chunk_pos);
            let chunked_rm = patch.wire_node(
                format!("{}.blockified", consumer.name),
                AxisOp::Rm(new_axis),
                &[chunked_term],
            )?[0];
            return Ok(Some((OutletId::new(consumer.id, 0), chunked_rm)));
        }
    }
    Ok(Some((OutletId::new(node.id, 0), chunked_term)))
}

/// EinSum terminator (e.g. ex02's `attn @ V`): chunkifies the second
/// EinSum the same way as the initiator.  Inputs already in `chunked`
/// (the multi-T-axis input from the body) are reused as-is; auxiliary
/// inputs (single-T-axis) get a tap + split reshape inserted.
fn wire_terminator_einsum(
    patch: &mut TypedModelPatch,
    model: &TypedModel,
    node: &TypedNode,
    op: &EinSum,
    chunked: &HashMap<OutletId, OutletId>,
    chunk_sym: &Symbol,
    k: i64,
) -> TractResult<Option<(OutletId, OutletId)>> {
    let mut chunked_inputs: TVec<OutletId> = tvec!();
    let mut input_starts: Vec<Option<usize>> = vec![];
    for (slot, &input) in node.inputs.iter().enumerate() {
        let positions = streaming_positions(model.outlet_fact(input)?, chunk_sym);
        if let Some(&already_chunked) = chunked.get(&input) {
            // Multi-T-axis input from the body — already in chunked form.
            chunked_inputs.push(already_chunked);
            input_starts.push(positions.first().copied());
        } else if positions.len() == 1 {
            let tapped = patch.tap_model(model, input)?;
            let in_fact = patch.outlet_fact(tapped)?.clone();
            let stream_axis = in_fact
                .shape
                .iter()
                .position(|d| d.symbols().contains(chunk_sym))
                .ok_or_else(|| format_err!("auxiliary input lost streaming axis"))?;
            let from = tvec!(in_fact.shape[stream_axis].clone());
            let to = tvec!(chunk_sym.to_dim(), k.to_dim());
            let reshape = AxisOp::Reshape(stream_axis, from, to);
            let new_chunked = patch.wire_node(
                format!("{}.blockify_split.in{slot}", node.name),
                reshape,
                &[tapped],
            )?[0];
            chunked_inputs.push(new_chunked);
            input_starts.push(Some(positions[0]));
        } else if positions.is_empty() {
            chunked_inputs.push(patch.tap_model(model, input)?);
            input_starts.push(None);
        } else {
            bail!(
                "Blockify: EinSum terminator input {slot} has {} streaming axes (max 2)",
                positions.len()
            );
        }
    }

    let out_streaming = streaming_positions(&node.outputs[0].fact, chunk_sym);
    let chunked_op = chunkify_einsum(op, &input_starts, out_streaming.first().copied())?;
    let chunked_term =
        patch.wire_node(format!("{}.blockified", node.name), chunked_op, &chunked_inputs)?[0];
    Ok(Some((OutletId::new(node.id, 0), chunked_term)))
}

/// Wire the boundary merge reshape: collapses [..., S, k, ...] back to
/// [..., k·S, ...] so the patch's output matches the original outlet's
/// shape (which is [..., k·S, ...] post-substitution).
fn wire_merge_reshape(
    patch: &mut TypedModelPatch,
    boundary_name: &str,
    chunked_form: OutletId,
    chunk_sym: &Symbol,
    k: i64,
) -> TractResult<OutletId> {
    let chunked_fact = patch.outlet_fact(chunked_form)?.clone();
    let chunk_pos = chunked_fact.shape.iter().position(|d| d == &chunk_sym.to_dim());
    if let Some(pos) = chunk_pos
        && pos + 1 < chunked_fact.shape.len()
        && chunked_fact.shape[pos + 1] == k.to_dim()
    {
        let from = tvec!(chunk_sym.to_dim(), k.to_dim());
        let to = tvec!(chunk_sym.to_dim() * k);
        let reshape = AxisOp::Reshape(pos, from, to);
        Ok(patch.wire_node(
            format!("{}.blockify_merge", boundary_name),
            reshape,
            &[chunked_form],
        )?[0])
    } else {
        Ok(chunked_form)
    }
}

/// First streaming-symbol-bearing symbol on a fact's shape.  Used by
/// terminator wiring to derive the chunk insertion position from the
/// input fact, op-agnostically.
fn first_streaming_symbol(fact: &TypedFact) -> Option<Symbol> {
    fact.shape.iter().find_map(|d| d.symbols().into_iter().next())
}

/// Op-agnostic dead-node identification: section nodes whose output has
/// `uniform_tdim` (mask construction), plus any node whose outlets are
/// either obliterated by an existing shunt or whose only consumers are
/// dead.  Shunted boundary nodes are also dead (their original output
/// is no longer reachable from any model output).
fn collect_dead_nodes(
    model: &TypedModel,
    sec: &QuadraticSection,
    shunts: &HashMap<OutletId, OutletId>,
) -> Vec<usize> {
    // Model inputs are never marked dead — they're still produced by
    // upstream code and may feed nodes outside the section.
    let inputs: HashSet<usize> =
        model.input_outlets().map(|outs| outs.iter().map(|o| o.node).collect()).unwrap_or_default();
    // Seed: every section node (initiators, body, mask-construction
    // wires whose only consumers are body Mul-by-mask) — they're all
    // replaced by chunked patch nodes.
    let mut dead: HashSet<usize> = sec.section.iter().copied().collect();
    // Boundary nodes (shunted outlets' source nodes) are dead.
    for shunted in shunts.keys() {
        dead.insert(shunted.node);
    }
    // Walk back: any node whose only consumers are dead is also dead.
    loop {
        let mut changed = false;
        for n in &model.nodes {
            if dead.contains(&n.id) || inputs.contains(&n.id) {
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
    dead.into_iter().collect()
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
    let new_repr = op.axes.available_label();
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
        let chunk_char = op.axes.available_label();
        assert_eq!(ins[0], format!("{chunk_char}id"));
        assert_eq!(ins[1], format!("{chunk_char}jd"));
        assert_eq!(outs[0], format!("{chunk_char}ij"));
    }

    #[test]
    fn chunkify_einsum_handles_streaming_at_inner_position() {
        let op = einsum_for(&["bid", "bjd"], "bij");
        let chunked = ck(&op, &[1, 1], 1);
        let (ins, outs) = axes_to_strings(&chunked);
        let chunk_char = op.axes.available_label();
        assert_eq!(ins[0], format!("b{chunk_char}id"));
        assert_eq!(ins[1], format!("b{chunk_char}jd"));
        assert_eq!(outs[0], format!("b{chunk_char}ij"));
    }

    #[test]
    fn chunkify_einsum_handles_mixed_input_positions() {
        let op = einsum_for(&["id", "bjd"], "bij");
        let chunked = ck(&op, &[0, 1], 1);
        let (ins, outs) = axes_to_strings(&chunked);
        let chunk_char = op.axes.available_label();
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
        let chunk_char = op.axes.available_label();
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
