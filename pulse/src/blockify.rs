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
pub fn blockify(
    mut model: TypedModel,
    stream_sym: Symbol,
    pulse: TDim,
) -> TractResult<BlockifyResult> {
    let Some(found) = find_pattern(&model, &stream_sym)? else {
        return Ok(BlockifyResult { model, stream_sym, pulse });
    };
    let (new_model, chunk_sym, translated_pulse) = rewrite(&model, &stream_sym, &found, &pulse)?;
    model = new_model;
    Ok(BlockifyResult { model, stream_sym: chunk_sym, pulse: translated_pulse })
}

/// A connected subgraph of the typed model where every wire has multi-T-axis
/// shape (≥2 streaming-symbol axes), bracketed by single-T-axis wires.
///
/// Phase 1+2+3 of Blockify recognition produces this structure op-agnostically.
/// Phase 4 (the rewrite) consumes it and dispatches per op-type.
#[derive(Debug)]
struct QuadraticSection {
    /// All nodes whose output wire has multi-T-axis shape.  Today the bridge
    /// to `Pattern` walks initiators/terminators only; the full set is here
    /// because phase 4 generalisation (body-chain handling) will need it.
    #[allow(dead_code)]
    section: std::collections::BTreeSet<usize>,
    /// Subset of `section` whose inputs are all outside it (= "rise to quadratic").
    initiators: Vec<usize>,
    /// Nodes outside `section` consuming an in-section wire (= "drop back to linear").
    terminators: Vec<usize>,
    /// Block size extracted from a recognisable mask in the section.
    chunk_size: i64,
}

/// Phase 1+2+3: detect a section of the graph where wires go multi-T-axis,
/// verify every wire there has structural justification (uniform_tdim or
/// region_of_interest), and confirm at least one wire has a recognisable
/// mask form so we know the chunk size.
fn find_quadratic_section(
    model: &TypedModel,
    stream_sym: &Symbol,
) -> TractResult<Option<QuadraticSection>> {
    use std::collections::BTreeSet;

    let is_multi_t_axis = |fact: &TypedFact| {
        fact.shape.iter().filter(|d| d.symbols().contains(stream_sym)).count() >= 2
    };

    // Phase 1 — topology.
    let mut section: BTreeSet<usize> = BTreeSet::new();
    for node in &model.nodes {
        if node.outputs.len() != 1 {
            continue;
        }
        if is_multi_t_axis(&node.outputs[0].fact) {
            section.insert(node.id);
        }
    }
    if section.is_empty() {
        return Ok(None);
    }

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

    // Phase 2 — structural-justification coverage.  At least one wire in the
    // section must carry uniform_tdim or region_of_interest; that's the
    // anchor we'll use in phase 3 to find the chunk size.  Wires without
    // either are tolerated as long as their effective values are bounded
    // by an upstream mask multiplication (the typical case for the score
    // matrix wire post-Mul).
    let any_annotated = section.iter().any(|&nid| {
        let fact = &model.nodes[nid].outputs[0].fact;
        fact.uniform_tdim.is_some() || fact.region_of_interest.is_some()
    });
    if !any_annotated {
        return Ok(None);
    }

    // Phase 3 — form recognition.  Find at least one wire whose uniform_tdim
    // matches the block-diagonal form `chunk(i) == chunk(j)`; chunk size
    // determines the per-chunk dimension.
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
        return Ok(None);
    };

    Ok(Some(QuadraticSection { section, initiators, terminators, chunk_size }))
}

/// Wires identified by the recogniser as the rewrite target.
///
/// Op-aware bridge from a `QuadraticSection` (phase 1+2+3) to the rewriter
/// (phase 4).  Identifies which initiator is the compute EinSum, which body
/// node is the Mul-by-mask, and which terminator is the Reduce<Sum>, plus
/// the streaming-axis positions the rewriter needs.
#[derive(Debug)]
struct Pattern {
    einsum_node: usize,
    mul_node: usize,
    /// The wire that's the mask (Mul's other input).
    mask_outlet: OutletId,
    reduce_node: usize,
    /// Block size extracted from the mask's uniform_tdim.
    chunk_size: i64,
    /// Position of the streaming axis on each EinSum input, in input order.
    einsum_in_streaming_axes: TVec<usize>,
    /// Positions of the streaming axes on the EinSum's output.  Required to
    /// be contiguous (the chunk batch axis is inserted at the first one and
    /// the second one becomes within-chunk on the next slot).
    einsum_out_streaming_axes: TVec<usize>,
    /// Reduce axis (on Mul/EinSum output).
    reduce_axis: usize,
}

impl Pattern {
    /// Phase 4 prep: derive op-specific rewrite info from a topological
    /// section.  Returns None if the section's op-types aren't ones the
    /// current rewriter handles.
    fn from_section(
        model: &TypedModel,
        sec: &QuadraticSection,
        stream_sym: &Symbol,
    ) -> TractResult<Option<Pattern>> {
        // Initiator: a section node that's a compute EinSum (output has no
        // uniform_tdim, op is EinSum).  Mask-construction initiators (Eq,
        // etc.) have uniform_tdim outputs — they get deleted, not rewritten.
        let einsum_node = sec.initiators.iter().copied().find(|&nid| {
            let n = &model.nodes[nid];
            n.outputs[0].fact.uniform_tdim.is_none() && n.op_is::<EinSum>()
        });
        let Some(einsum_node) = einsum_node else {
            return Ok(None);
        };
        let einsum_n = &model.nodes[einsum_node];
        let einsum_out_fact = &einsum_n.outputs[0].fact;
        let einsum_out_streaming_axes: TVec<usize> = einsum_out_fact
            .shape
            .iter()
            .enumerate()
            .filter(|(_, d)| d.symbols().contains(stream_sym))
            .map(|(i, _)| i)
            .collect();
        if einsum_out_streaming_axes.len() != 2
            || einsum_out_streaming_axes[1] != einsum_out_streaming_axes[0] + 1
        {
            return Ok(None);
        }
        let mut einsum_in_streaming_axes: TVec<usize> = tvec!();
        for &input in &einsum_n.inputs {
            let f = model.outlet_fact(input)?;
            let positions: TVec<usize> = f
                .shape
                .iter()
                .enumerate()
                .filter(|(_, d)| d.symbols().contains(stream_sym))
                .map(|(i, _)| i)
                .collect();
            if positions.len() != 1 {
                return Ok(None);
            }
            einsum_in_streaming_axes.push(positions[0]);
        }

        // Body Mul-by-mask: a section node consuming the EinSum's output and
        // a uniform_tdim wire.  Identify the mask_outlet as the uniform_tdim
        // input of that Mul.
        let einsum_consumers = model.outlet_successors(OutletId::new(einsum_node, 0));
        if einsum_consumers.len() != 1 {
            return Ok(None);
        }
        let mul_node = einsum_consumers[0].node;
        let mul_n = &model.nodes[mul_node];
        let Some(bin) = mul_n.op_as::<TypedBinOp>() else {
            return Ok(None);
        };
        if !bin.0.is::<Mul>() {
            return Ok(None);
        }
        let mask_outlet = mul_n
            .inputs
            .iter()
            .copied()
            .find(|i| model.outlet_fact(*i).map(|f| f.uniform_tdim.is_some()).unwrap_or(false))
            .ok_or_else(|| format_err!("Mul body node has no uniform_tdim input"))?;

        // Terminator Reduce<Sum>: a terminator op whose op is Reduce<Sum>
        // on a streaming axis.
        let reduce_node = sec.terminators.iter().copied().find(|&nid| {
            let n = &model.nodes[nid];
            if let Some(r) = n.op_as::<Reduce>() {
                r.reducer == Reducer::Sum
                    && r.axes.len() == 1
                    && einsum_out_streaming_axes.contains(&r.axes[0])
            } else {
                false
            }
        });
        let Some(reduce_node) = reduce_node else {
            return Ok(None);
        };
        let reduce_axis = model.nodes[reduce_node]
            .op_as::<Reduce>()
            .ok_or_else(|| format_err!("expected Reduce"))?
            .axes[0];

        Ok(Some(Pattern {
            einsum_node,
            mul_node,
            mask_outlet,
            reduce_node,
            chunk_size: sec.chunk_size,
            einsum_in_streaming_axes,
            einsum_out_streaming_axes,
            reduce_axis,
        }))
    }
}

fn find_pattern(model: &TypedModel, stream_sym: &Symbol) -> TractResult<Option<Pattern>> {
    let Some(sec) = find_quadratic_section(model, stream_sym)? else {
        return Ok(None);
    };
    Pattern::from_section(model, &sec, stream_sym)
}

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

/// Rebuild the model: substitute T → k·S throughout, replace the rewrite
/// region with the chunked equivalent.
fn rewrite(
    model: &TypedModel,
    stream_sym: &Symbol,
    pat: &Pattern,
    pulse: &TDim,
) -> TractResult<(TypedModel, Symbol, TDim)> {
    let mut out = TypedModel::default();
    out.symbols = model.symbols.clone();
    let chunk_sym = out.symbols.new_with_prefix("S");
    let k = pat.chunk_size;
    let chunk_dim: TDim = chunk_sym.to_dim() * k;

    // T → k·S on every TDim we copy.
    let subst = |d: &TDim| -> TractResult<TDim> { d.substitute(stream_sym, &chunk_dim) };
    let subst_shape = |s: &ShapeFact| -> TractResult<ShapeFact> {
        let dims: TractResult<TVec<TDim>> = s.iter().map(|d| subst(&d)).collect();
        Ok(dims?.into())
    };

    // Identify the dead mask-construction subgraph.  Its only escape is
    // the mask outlet feeding the Mul, plus the model output (which we
    // rebuild ourselves at the end), so anything that only reaches the
    // model via that outlet can be skipped.
    let mut dead = std::collections::HashSet::<usize>::new();
    collect_dead_mask_nodes(model, pat, &mut dead);
    let mut skipped: std::collections::HashSet<usize> =
        [pat.einsum_node, pat.mul_node, pat.reduce_node].into_iter().collect();
    // Reduce's direct consumers depend on `reduce_out`, which we wire only
    // after the topological pass.  Skip them here; we'll wire them then.
    for cons in model.outlet_successors(OutletId::new(pat.reduce_node, 0)) {
        skipped.insert(cons.node);
    }

    // Map old OutletId → new OutletId.
    let mut mapping: HashMap<OutletId, OutletId> = HashMap::default();

    let order = model.eval_order()?;
    for &nid in &order {
        let node = &model.nodes[nid];
        if dead.contains(&nid) || skipped.contains(&nid) {
            continue;
        }
        // Sources: rewrite shape with T-substitution.
        if node.op_is::<tract_core::ops::source::TypedSource>() {
            let fact = &node.outputs[0].fact;
            let new_fact = TypedFact {
                datum_type: fact.datum_type,
                shape: subst_shape(&fact.shape)?,
                ..fact.clone()
            };
            let new_id = out.add_source(node.name.clone(), new_fact)?;
            mapping.insert(OutletId::new(nid, 0), new_id);
            continue;
        }
        // Other passthrough nodes: copy op + inputs, let output_facts redo.
        let inputs: TVec<OutletId> = node
            .inputs
            .iter()
            .map(|i| {
                mapping.get(i).copied().ok_or_else(|| {
                    format_err!("Missing mapping for input {i:?} of node {}", node.name)
                })
            })
            .collect::<TractResult<_>>()?;
        let new_outputs = out.wire_node(node.name.clone(), node.op.clone(), &inputs)?;
        for (slot, &out_id) in new_outputs.iter().enumerate() {
            mapping.insert(OutletId::new(nid, slot), out_id);
        }
    }

    // Now wire the rewrite region.
    let einsum_node = &model.nodes[pat.einsum_node];
    let einsum_op = einsum_node.op_as::<EinSum>().unwrap();
    let einsum_inputs: TVec<OutletId> = einsum_node.inputs.iter().map(|i| mapping[i]).collect();

    // For each EinSum input, insert a Reshape that splits the streaming axis
    // (which has dim k·S) into [S, k].
    let mut chunked_inputs: TVec<OutletId> = tvec!();
    for (ix, &input_outlet) in einsum_inputs.iter().enumerate() {
        let in_fact = out.outlet_fact(input_outlet)?.clone();
        // Find the streaming axis in this input.
        let stream_axis =
            in_fact.shape.iter().position(|d| d.symbols().contains(&chunk_sym)).ok_or_else(
                || format_err!("EinSum input {ix} has no streaming axis after substitute"),
            )?;
        let from = tvec!(in_fact.shape[stream_axis].clone()); // = k·S
        let to = tvec!(chunk_sym.to_dim(), k.to_dim());
        let reshape = AxisOp::Reshape(stream_axis, from, to);
        let chunked = out.wire_node(
            format!("{}.blockify_split.{ix}", einsum_node.name),
            reshape,
            &[input_outlet],
        )?[0];
        chunked_inputs.push(chunked);
    }

    // Rewrite EinSum subscript: insert a chunk batch axis at the position
    // of the original streaming axis on each input, plus on the output for
    // each surviving streaming axis.
    let _ = stream_sym;
    let new_einsum = chunkify_einsum(einsum_op, pat)?;
    let new_einsum_out = out.wire_node(einsum_node.name.clone(), new_einsum, &chunked_inputs)?[0];

    // Skip the Mul: its "result" (under the chunk batch axis) is just the EinSum.
    // Rewrite Reduce axis through the inserted chunk batch axis.
    let reduce_node = &model.nodes[pat.reduce_node];
    let reduce_op = reduce_node.op_as::<Reduce>().unwrap();
    let new_reduce_axis = chunked_axis_index(reduce_op.axes[0], pat)?;
    let new_reduce = Reduce { axes: tvec!(new_reduce_axis), reducer: reduce_op.reducer };
    let reduce_out = out.wire_node(reduce_node.name.clone(), new_reduce, &[new_einsum_out])?[0];
    mapping.insert(OutletId::new(pat.reduce_node, 0), reduce_out);

    // Wire any nodes that consumed the Reduce output (e.g. Squeeze axis 0).
    // We already skipped reduce; everything depending on it must now be
    // routed through reduce_out, with its axis indices possibly shifted.
    // For the POC: we lazily emit any direct consumer of reduce that's a
    // Squeeze/RmAxis on the original reduce_axis, shifting it.
    // Otherwise we fall back to copying as-is (shape will be wrong → caught
    // by output_facts).
    let reduce_consumers: TVec<_> =
        model.outlet_successors(OutletId::new(pat.reduce_node, 0)).iter().copied().collect();
    for cons in &reduce_consumers {
        let cons_node = &model.nodes[cons.node];
        if let Some(AxisOp::Rm(axis)) = cons_node.op_as::<AxisOp>() {
            if *axis == pat.reduce_axis {
                let shifted = AxisOp::Rm(new_reduce_axis);
                let new_out = out.wire_node(cons_node.name.clone(), shifted, &[reduce_out])?[0];
                mapping.insert(OutletId::new(cons.node, 0), new_out);
                continue;
            }
        }
        // Fallback: copy op verbatim, hope its output_facts work after
        // shape changes.  POC limitation; will be revisited.
        let inputs: TVec<OutletId> = cons_node.inputs.iter().map(|i| mapping[i]).collect();
        let new_outputs = out.wire_node(cons_node.name.clone(), cons_node.op.clone(), &inputs)?;
        for (slot, &out_id) in new_outputs.iter().enumerate() {
            mapping.insert(OutletId::new(cons.node, slot), out_id);
        }
    }

    // For each model output: if its shape contains the chunk axis followed
    // immediately by a within-chunk axis of size k, flatten them back into
    // a single k·S axis at the same position.  This restores the original
    // streaming-axis layout without changing the model's external interface.
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
    out.select_output_outlets(&new_outputs)?;

    // Translate the user's pulse value from token-units to chunk-units.
    // Pulse on T (= k·S) of value V means V tokens per pulse; equivalently
    // V/k chunks per pulse on S.  Use TDim division.
    let translated_pulse =
        pulse.clone().maybe_div(&k.to_dim()).map(|(d, _)| d).map_err(|e| {
            format_err!("Blockify: pulse {pulse} not divisible by chunk size {k}: {e}")
        })?;
    Ok((out, chunk_sym, translated_pulse))
}

fn collect_dead_mask_nodes(
    model: &TypedModel,
    pat: &Pattern,
    dead: &mut std::collections::HashSet<usize>,
) {
    // Walk back from the mask outlet.  Anything whose only consumer is
    // (transitively) the mul_node is dead.
    let mut stack = vec![pat.mask_outlet.node];
    while let Some(nid) = stack.pop() {
        if dead.contains(&nid) {
            continue;
        }
        if nid == pat.einsum_node {
            continue;
        }
        // Check all consumers of this node's outputs are either the Mul
        // (the mask consumer) or already dead.
        let all_consumers_dead_or_mul = (0..model.nodes[nid].outputs.len()).all(|slot| {
            model
                .outlet_successors(OutletId::new(nid, slot))
                .iter()
                .all(|c| c.node == pat.mul_node || dead.contains(&c.node))
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

fn chunked_axis_index(orig_axis: usize, pat: &Pattern) -> TractResult<usize> {
    // The chunk batch axis is inserted at the position of the first
    // streaming output axis; every original axis at or after that position
    // shifts right by one.  Streaming axes themselves (the within-chunk
    // versions) are no exception — they shift by one too, because the
    // chunk axis lives where they used to start.
    let chunk_pos = pat.einsum_out_streaming_axes[0];
    if orig_axis < chunk_pos { Ok(orig_axis) } else { Ok(orig_axis + 1) }
}

/// Insert the new chunk axis character at the streaming-axis position in
/// every input subscript and at the first streaming-axis position in the
/// output subscript.  Within-chunk versions of the formerly-streaming axes
/// keep their original chars (now sitting at position+1 after the insert).
fn chunkify_einsum(op: &EinSum, pat: &Pattern) -> TractResult<EinSum> {
    let (inputs, outputs) = op.axes.to_strs();
    let new_repr = pick_free_axis_repr(&op.axes);
    let insert_at = |s: &String, pos: usize| -> String {
        let mut chars: Vec<char> = s.chars().collect();
        chars.insert(pos, new_repr);
        chars.into_iter().collect()
    };
    let new_inputs: Vec<String> = inputs
        .iter()
        .zip(pat.einsum_in_streaming_axes.iter())
        .map(|(s, &pos)| insert_at(s, pos))
        .collect();
    let chunk_pos = pat.einsum_out_streaming_axes[0];
    let new_outputs: Vec<String> = outputs
        .iter()
        .enumerate()
        .map(|(i, s)| if i == 0 { insert_at(s, chunk_pos) } else { s.clone() })
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

    fn make_pattern(in_axes: &[usize], out_axes: &[usize]) -> Pattern {
        Pattern {
            einsum_node: 0,
            mul_node: 0,
            mask_outlet: OutletId::new(0, 0),
            reduce_node: 0,
            chunk_size: 2,
            einsum_in_streaming_axes: in_axes.iter().copied().collect(),
            einsum_out_streaming_axes: out_axes.iter().copied().collect(),
            reduce_axis: 0,
        }
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

    #[test]
    fn chunkify_einsum_handles_streaming_at_position_zero() {
        // The ex01 case: "id,jd->ij" with streaming at position 0 on both
        // inputs and on output positions 0 and 1.
        let op = einsum_for(&["id", "jd"], "ij");
        let pat = make_pattern(&[0, 0], &[0, 1]);
        let chunked = chunkify_einsum(&op, &pat).unwrap();
        let (ins, outs) = axes_to_strings(&chunked);
        // The free char picked is the lowest unused letter.
        let chunk_char = pick_free_axis_repr(&op.axes);
        assert_eq!(ins[0], format!("{chunk_char}id"));
        assert_eq!(ins[1], format!("{chunk_char}jd"));
        assert_eq!(outs[0], format!("{chunk_char}ij"));
    }

    #[test]
    fn chunkify_einsum_handles_streaming_at_inner_position() {
        // Multi-head-like: "bid,bjd->bij" — streaming at position 1 on
        // inputs, positions 1 and 2 on output, batch axis b at 0.
        let op = einsum_for(&["bid", "bjd"], "bij");
        let pat = make_pattern(&[1, 1], &[1, 2]);
        let chunked = chunkify_einsum(&op, &pat).unwrap();
        let (ins, outs) = axes_to_strings(&chunked);
        let chunk_char = pick_free_axis_repr(&op.axes);
        assert_eq!(ins[0], format!("b{chunk_char}id"));
        assert_eq!(ins[1], format!("b{chunk_char}jd"));
        assert_eq!(outs[0], format!("b{chunk_char}ij"));
    }

    #[test]
    fn chunkify_einsum_handles_mixed_input_positions() {
        // Streaming at different positions on the two inputs:
        // "id,bjd->bij" — input 0 has streaming at pos 0,
        // input 1 has streaming at pos 1.  (Output i, j at pos 1, 2.)
        let op = einsum_for(&["id", "bjd"], "bij");
        let pat = make_pattern(&[0, 1], &[1, 2]);
        let chunked = chunkify_einsum(&op, &pat).unwrap();
        let (ins, outs) = axes_to_strings(&chunked);
        let chunk_char = pick_free_axis_repr(&op.axes);
        assert_eq!(ins[0], format!("{chunk_char}id"));
        assert_eq!(ins[1], format!("b{chunk_char}jd"));
        assert_eq!(outs[0], format!("b{chunk_char}ij"));
    }

    #[test]
    fn chunked_axis_index_zero_chunk_position() {
        let pat = make_pattern(&[0, 0], &[0, 1]);
        // Chunk at output pos 0; everything shifts right by 1.
        assert_eq!(chunked_axis_index(0, &pat).unwrap(), 1);
        assert_eq!(chunked_axis_index(1, &pat).unwrap(), 2);
    }

    #[test]
    fn chunked_axis_index_inner_chunk_position() {
        // Chunk at output pos 1; axis 0 stays, axes 1+ shift right.
        let pat = make_pattern(&[1, 1], &[1, 2]);
        assert_eq!(chunked_axis_index(0, &pat).unwrap(), 0);
        assert_eq!(chunked_axis_index(1, &pat).unwrap(), 2);
        assert_eq!(chunked_axis_index(2, &pat).unwrap(), 3);
    }
}
