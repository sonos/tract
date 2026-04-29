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
use std::collections::HashMap;
use tract_core::axes::AxesMapping;
use tract_core::model::TypedModelPatch;
use tract_core::ops::binary::TypedBinOp;
use tract_core::ops::change_axes::AxisOp;
use tract_core::ops::einsum::EinSum;
use tract_core::ops::math::Mul;
use tract_core::ops::nn::{Reduce, Reducer};

/// Result of running Blockify: the (possibly-rewritten) model, plus the
/// streaming symbol and pulse value to use downstream.  Blockify substitutes
/// the user's stream symbol `T` with `k*S` in the rewritten subgraph;
/// downstream pulsification must use `S` and a translated pulse value.
pub struct BlockifyResult {
    pub model: TypedModel,
    pub stream_sym: Symbol,
    pub pulse: TDim,
}

/// Find every quadratic section in `model`, substitute T → k·S globally,
/// then apply one TypedModelPatch per section that chunkifies it.  No-op
/// if no recognisable section is found.
pub fn blockify(model: TypedModel, stream_sym: Symbol, pulse: TDim) -> TractResult<BlockifyResult> {
    let sections = find_quadratic_sections(&model, &stream_sym)?;
    if sections.is_empty() {
        return Ok(BlockifyResult { model, stream_sym, pulse });
    }
    let k = sections[0].chunk_size;
    if !sections.iter().all(|s| s.chunk_size == k) {
        bail!(
            "Blockify found multiple quadratic sections with mismatched chunk sizes; \
             a single global symbol substitution cannot cover them.  Refusing to \
             blockify rather than produce a partial rewrite."
        );
    }

    // Phase 1: introduce S, substitute T → k·S globally via core.
    let chunk_sym = model.symbols.new_with_prefix("S");
    let subs: HashMap<Symbol, TDim> = HashMap::from([(stream_sym.clone(), chunk_sym.to_dim() * k)]);
    let mut model = model.substitute_symbols(&subs)?;

    // Phase 2: one TypedModelPatch per section.  Section node ids are
    // stable across `substitute_symbols` (1-to-1 model copy), so we can
    // reuse the sections we detected on the original.
    for sec in &sections {
        let Some(patch) = build_section_patch(&model, sec, &chunk_sym, k)? else {
            // Op-types in this section aren't handled; leave it.  Pulsification
            // will fail downstream with a recoverable error if this matters.
            continue;
        };
        patch.apply(&mut model)?;
    }

    let translated_pulse =
        pulse.clone().maybe_div(&k.to_dim()).map(|(d, _)| d).map_err(|e| {
            format_err!("Blockify: pulse {pulse} not divisible by chunk size {k}: {e}")
        })?;
    Ok(BlockifyResult { model, stream_sym: chunk_sym, pulse: translated_pulse })
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

/// Build a TypedModelPatch that chunkifies one quadratic section.  Returns
/// `Ok(None)` if the section's op-types aren't handled (the section is
/// left alone; downstream pulsification will fail with a clear error).
fn build_section_patch(
    model: &TypedModel,
    sec: &QuadraticSection,
    chunk_sym: &Symbol,
    k: i64,
) -> TractResult<Option<TypedModelPatch>> {
    // ── Identify the section's per-role nodes ─────────────────────────────
    //
    // Compute initiator: an EinSum in `sec.initiators` whose output has no
    // uniform_tdim (mask-construction initiators are dead, not chunkified).
    let einsum_node_id = sec.initiators.iter().copied().find(|&nid| {
        let n = &model.nodes[nid];
        n.outputs[0].fact.uniform_tdim.is_none() && n.op_is::<EinSum>()
    });
    let Some(einsum_node_id) = einsum_node_id else {
        return Ok(None);
    };
    let einsum_node = &model.nodes[einsum_node_id];
    let einsum_op = einsum_node.op_as::<EinSum>().unwrap();

    let einsum_out_streaming_axes = streaming_positions(&einsum_node.outputs[0].fact, chunk_sym);
    if einsum_out_streaming_axes.len() != 2
        || einsum_out_streaming_axes[1] != einsum_out_streaming_axes[0] + 1
    {
        return Ok(None);
    }
    let mut einsum_in_streaming_axes: TVec<usize> = tvec!();
    for &input in &einsum_node.inputs {
        let positions = streaming_positions(model.outlet_fact(input)?, chunk_sym);
        if positions.len() != 1 {
            return Ok(None);
        }
        einsum_in_streaming_axes.push(positions[0]);
    }

    // Body Mul-by-mask: the EinSum's single consumer must be a Mul with
    // a uniform_tdim auxiliary input.
    let einsum_consumers = model.outlet_successors(OutletId::new(einsum_node_id, 0));
    if einsum_consumers.len() != 1 {
        return Ok(None);
    }
    let mul_node_id = einsum_consumers[0].node;
    let mul_node = &model.nodes[mul_node_id];
    let Some(bin) = mul_node.op_as::<TypedBinOp>() else {
        return Ok(None);
    };
    if !bin.0.is::<Mul>() {
        return Ok(None);
    }
    let Some(mask_outlet) = mul_node
        .inputs
        .iter()
        .copied()
        .find(|i| model.outlet_fact(*i).map(|f| f.uniform_tdim.is_some()).unwrap_or(false))
    else {
        return Ok(None);
    };

    // Terminator: Reduce<Sum> on a streaming axis, or contracting EinSum.
    let terminator_node_id = sec.terminators.iter().copied().find(|&nid| {
        let n = &model.nodes[nid];
        if let Some(r) = n.op_as::<Reduce>() {
            return r.reducer == Reducer::Sum
                && r.axes.len() == 1
                && einsum_out_streaming_axes.contains(&r.axes[0]);
        }
        if let Some(es) = n.op_as::<EinSum>() {
            let Some(mt_slot) = n.inputs.iter().position(|i| i.node == mul_node_id) else {
                return false;
            };
            let (in_strs, out_strs) = es.axes.to_strs();
            let Some(mt_subscript) = in_strs.get(mt_slot) else {
                return false;
            };
            let stream_chars: Vec<char> = einsum_out_streaming_axes
                .iter()
                .map(|&pos| mt_subscript.chars().nth(pos))
                .collect::<Option<Vec<_>>>()
                .unwrap_or_default();
            if stream_chars.is_empty() {
                return false;
            }
            return stream_chars.iter().any(|c| !out_strs[0].contains(*c));
        }
        false
    });
    let Some(terminator_node_id) = terminator_node_id else {
        return Ok(None);
    };
    let term_node = &model.nodes[terminator_node_id];
    let chunk_axis_in_einsum_output = einsum_out_streaming_axes[0];

    // ── Build the patch ───────────────────────────────────────────────────

    let mut patch = TypedModelPatch::default();

    // 1. Tap each EinSum input from the model and split it via reshape.
    let mut chunked_einsum_inputs: TVec<OutletId> = tvec!();
    for (ix, (&input, &stream_axis)) in
        einsum_node.inputs.iter().zip(einsum_in_streaming_axes.iter()).enumerate()
    {
        let tapped = patch.tap_model(model, input)?;
        let in_fact = patch.outlet_fact(tapped)?.clone();
        let from = tvec!(in_fact.shape[stream_axis].clone());
        let to = tvec!(chunk_sym.to_dim(), k.to_dim());
        let reshape = AxisOp::Reshape(stream_axis, from, to);
        let chunked = patch.wire_node(
            format!("{}.blockify_split.{ix}", einsum_node.name),
            reshape,
            &[tapped],
        )?[0];
        chunked_einsum_inputs.push(chunked);
    }

    // 2. Wire the chunked initiator EinSum.
    let in_starts: Vec<Option<usize>> = einsum_in_streaming_axes.iter().map(|&p| Some(p)).collect();
    let chunked_einsum_op =
        chunkify_einsum(einsum_op, &in_starts, Some(chunk_axis_in_einsum_output))?;
    let chunked_einsum_out = patch.wire_node(
        format!("{}.blockified", einsum_node.name),
        chunked_einsum_op,
        &chunked_einsum_inputs,
    )?[0];

    // 3. Wire the chunked terminator.  The body Mul-by-mask's chunked
    //    output is the chunked initiator output (mask is all-1 in chunked form).
    let chunked_term_out = if let Some(reduce_op) = term_node.op_as::<Reduce>() {
        let new_axis = chunked_axis_index(reduce_op.axes[0], chunk_axis_in_einsum_output);
        let new_reduce = Reduce { axes: tvec!(new_axis), reducer: reduce_op.reducer };
        patch.wire_node(
            format!("{}.blockified", term_node.name),
            new_reduce,
            &[chunked_einsum_out],
        )?[0]
    } else if let Some(es_op) = term_node.op_as::<EinSum>() {
        let mt_slot = term_node
            .inputs
            .iter()
            .position(|i| i.node == mul_node_id)
            .ok_or_else(|| format_err!("EinSum terminator must consume the masked wire"))?;
        let mut chunked_inputs: TVec<OutletId> = tvec!();
        let mut input_starts: Vec<Option<usize>> = vec![];
        for (slot, &input) in term_node.inputs.iter().enumerate() {
            let positions = streaming_positions(model.outlet_fact(input)?, chunk_sym);
            if slot == mt_slot {
                chunked_inputs.push(chunked_einsum_out);
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
                let chunked = patch.wire_node(
                    format!("{}.blockify_split.in{slot}", term_node.name),
                    reshape,
                    &[tapped],
                )?[0];
                chunked_inputs.push(chunked);
                input_starts.push(Some(positions[0]));
            } else if positions.is_empty() {
                let tapped = patch.tap_model(model, input)?;
                chunked_inputs.push(tapped);
                input_starts.push(None);
            } else {
                bail!(
                    "Blockify: EinSum terminator input {slot} has {} streaming axes (max 2)",
                    positions.len()
                );
            }
        }
        let out_streaming = streaming_positions(&term_node.outputs[0].fact, chunk_sym);
        let chunked_term_op =
            chunkify_einsum(es_op, &input_starts, out_streaming.first().copied())?;
        patch.wire_node(
            format!("{}.blockified", term_node.name),
            chunked_term_op,
            &chunked_inputs,
        )?[0]
    } else {
        return Ok(None);
    };

    // 4. Identify the boundary outlet (where shapes match for shunt) and
    //    wire any post-terminator op (e.g. RmAxis on the reduce axis) plus
    //    the merge reshape.
    let (boundary_outlet, chunked_form) = if let Some(reduce_op) = term_node.op_as::<Reduce>() {
        let term_consumers = model.outlet_successors(OutletId::new(terminator_node_id, 0));
        if term_consumers.len() == 1 {
            let consumer = &model.nodes[term_consumers[0].node];
            if let Some(AxisOp::Rm(axis)) = consumer.op_as::<AxisOp>()
                && *axis == reduce_op.axes[0]
            {
                let new_axis = chunked_axis_index(reduce_op.axes[0], chunk_axis_in_einsum_output);
                let chunked_rm = patch.wire_node(
                    format!("{}.blockified", consumer.name),
                    AxisOp::Rm(new_axis),
                    &[chunked_term_out],
                )?[0];
                (OutletId::new(consumer.id, 0), chunked_rm)
            } else {
                (OutletId::new(terminator_node_id, 0), chunked_term_out)
            }
        } else {
            (OutletId::new(terminator_node_id, 0), chunked_term_out)
        }
    } else {
        (OutletId::new(terminator_node_id, 0), chunked_term_out)
    };

    // 5. Merge reshape: collapse [..., S, k, ...] back to [..., k·S, ...].
    let merged = {
        let chunked_fact = patch.outlet_fact(chunked_form)?.clone();
        let chunk_pos = chunked_fact.shape.iter().position(|d| d == &chunk_sym.to_dim());
        if let Some(pos) = chunk_pos
            && pos + 1 < chunked_fact.shape.len()
            && chunked_fact.shape[pos + 1] == k.to_dim()
        {
            let from = tvec!(chunk_sym.to_dim(), k.to_dim());
            let to = tvec!(chunk_sym.to_dim() * k);
            let reshape = AxisOp::Reshape(pos, from, to);
            patch.wire_node(
                format!("{}.blockify_merge", model.nodes[boundary_outlet.node].name),
                reshape,
                &[chunked_form],
            )?[0]
        } else {
            chunked_form
        }
    };

    // 6. Shunt the boundary outlet to the merged result.  Compatibility
    //    holds because the merge brought the chunked form back to the same
    //    [k·S, ...] shape the original outlet has post-substitution.
    patch.shunt_outside(model, boundary_outlet, merged)?;

    // 7. Obliterate the original section nodes that are no longer reachable
    //    from any model output.  The mask-construction subgraph plus the
    //    original initiator / body / terminator (and the post-terminator op
    //    that was rerouted) are dead post-shunt.
    patch.obliterate(einsum_node_id)?;
    patch.obliterate(mul_node_id)?;
    patch.obliterate(terminator_node_id)?;
    if boundary_outlet.node != terminator_node_id {
        patch.obliterate(boundary_outlet.node)?;
    }
    let mut mask_dead = std::collections::HashSet::<usize>::new();
    collect_dead_mask_nodes(model, mask_outlet.node, einsum_node_id, mul_node_id, &mut mask_dead);
    for nid in mask_dead {
        patch.obliterate(nid)?;
    }

    Ok(Some(patch))
}

fn collect_dead_mask_nodes(
    model: &TypedModel,
    mask_outlet_node: usize,
    einsum_node_id: usize,
    mul_node_id: usize,
    dead: &mut std::collections::HashSet<usize>,
) {
    let mut stack = vec![mask_outlet_node];
    while let Some(nid) = stack.pop() {
        if dead.contains(&nid) {
            continue;
        }
        if nid == einsum_node_id {
            continue;
        }
        let all_consumers_dead_or_mul = (0..model.nodes[nid].outputs.len()).all(|slot| {
            model
                .outlet_successors(OutletId::new(nid, slot))
                .iter()
                .all(|c| c.node == mul_node_id || dead.contains(&c.node))
        });
        if !all_consumers_dead_or_mul {
            continue;
        }
        dead.insert(nid);
        for inp in &model.nodes[nid].inputs {
            stack.push(inp.node);
        }
    }
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
