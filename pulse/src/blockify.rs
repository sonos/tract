//! Blockify — typed-model rewrite that factors block-diagonal / banded
//! structure into the graph topology, so the result has a single streaming
//! axis everywhere and pulsifies under v1's existing machinery.
//!
//! Recogniser scope: banded masks `chunk(axis_a) − chunk(axis_b) ∈ [lower, upper]`
//! (block-diagonal is the special case `lower == upper == 0`).
//!
//!   EinSum([a, b]) producing scores[T, T]
//!   → body op(s) consuming scores and the mask wire (whose
//!     `uniform_tdim` carries one of the recognised forms,
//!     `Eq(coord_a/k, coord_b/k)` or `Mul([Ge(upper, D), Ge(D, lower)])`
//!     with `D = coord_a/k − coord_b/k`)
//!   → Reduce<Sum> on a streaming axis  OR  contracting EinSum (ex02)
//!
//! Body ops are replayed in the chunked frame.  Each input is one of:
//!
//! * **chunked data** (in the `chunked` map): pass through.
//! * **mask wire** (source has `uniform_tdim` set): substitute a
//!   `[1, k_a, k_b]` constant block — the mask's value on the in-band
//!   region of every chunk.  For the recogniser's current scope the
//!   in-band value is uniformly `true`/`1`/`1.0` of the source dtype,
//!   so the const is a tile of that.  No per-op "mask role" knowledge:
//!   the body op just sees a regular tensor input it broadcasts over
//!   the chunk axis.  This is the user's "the score is a stream of
//!   constant-size blocks; the mask is a constant of the size of the
//!   block" reframing.
//! * **other external** (taps): rank-bumped with `AddAxis(0)` to
//!   match the chunked frame so rank-strict consumers (TypedBinOp, …)
//!   accept them.
//!
//! `axes_mapping::track_axis` asserts each chunked input's chunk axis
//! reaches a unique output axis — bails if the op would disconnect it.
//!
//! Implementation layout: detect quadratic sections globally, substitute
//! the streaming symbol T → k·S via core's `substitute_symbols`, then
//! build one TypedModelPatch per section that does the chunkification
//! locally and shunts the boundary outlet to the chunked-then-merged
//! output.  Sections are independent, so patches apply in sequence.
//!
//! Banded sections wrap any input on the contracted-axis side with
//! `WindowOnAxis(W) + flatten(W, k) → W·k` so the chunked einsum's
//! contracted within-chunk axis carries `W·k` rather than `k` elements.
//! "Contracted axis" = the score-matrix axis the terminator op contracts
//! (Reduce<Sum>'s reduced axis, or the EinSum axis missing from the
//! output).  Detected by `detect_contracted_score_axis`.
//!
//! Window slot offset:
//! * `contracted_axis == mask.axis_a`: `start = mask.lower` (consumer
//!   logical chunk c on the kept axis = axis_b, window covers
//!   `chunk(axis_a) ∈ [c + lower, c + upper]`).
//! * `contracted_axis == mask.axis_b`: `start = -mask.upper` (kept axis =
//!   axis_a, window covers `chunk(axis_b) ∈ [c - upper, c - lower]`).
//!
//! Output `stream.delay` = `max(0, end_of_window)` chunks (positive when
//! the window extends past `c`, zero when fully causal).
//!
//! For ex02-style EinSum terminators, auxiliary inputs whose stream axis
//! tracks (through the terminator einsum) to the contracted score axis
//! are also windowed, so all inputs to the terminator share the same
//! W·k contracted-axis size.
//!
//! `mask.lower > 0` (purely-future, skipping current) and `mask.upper < 0`
//! (purely-past) are rejected: they don't appear in attention masks and
//! would need different pulsifier wiring.

use crate::internal::*;
use std::collections::{BTreeSet, HashMap};
use tract_core::axes::AxesMapping;
use tract_core::model::TypedModelPatch;
use tract_core::ops::change_axes::AxisOp;
use tract_core::ops::einsum::EinSum;
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
        let k = sections[0].mask.chunk_size;
        if !sections.iter().all(|s| s.mask.chunk_size == k) {
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

        // Phase 2: one TypedModelPatch per section.  Sections the rewriter
        // can't fully handle (unknown body op, out-of-range mask form, …)
        // bail out of `build_section_patch` with a specific error: better
        // a clear Blockify failure here than a confusing pulsification
        // error a few stages later.
        for sec in &sections {
            let patch = build_section_patch(&new_model, sec, &chunk_sym, k)?;
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
    /// Mask form extracted from the section.  Determines the rewrite shape
    /// (block-diagonal vs banded) and carries the chunk size.
    mask: MaskForm,
    /// Score-matrix axis (in the mask's frame, 0 or 1) that is contracted
    /// by the terminator op.  For Reduce<Sum>, that's the reduced axis.
    /// For an EinSum terminator, it's the streaming axis of input 0 that
    /// doesn't appear in the output.  The windowed input(s) are those
    /// whose stream axis maps to this score axis.
    contracted_axis: usize,
}

/// Closed enum of mask forms the recogniser handles today.  All forms are a
/// banded predicate on `chunk(axis_a) - chunk(axis_b) ∈ [lower, upper]`;
/// the canonical block-diagonal mask is the special case `lower == upper == 0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MaskForm {
    chunk_size: i64,
    lower: i64,
    upper: i64,
    /// Axis whose chunk index appears with positive sign in the diff.
    axis_a: usize,
    /// Axis whose chunk index appears negated in the diff.
    axis_b: usize,
}

impl MaskForm {
    fn is_block_diag(&self) -> bool {
        self.lower == 0 && self.upper == 0
    }
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
        let mut mask: Option<MaskForm> = None;
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
            if let Some(form) = decode_mask(uniform, &streaming_axes) {
                mask = Some(form);
                break;
            }
        }
        let Some(mask) = mask else {
            continue;
        };

        // Phase 3b — find the score-matrix axis the terminator contracts.
        // All terminators of one section must agree (otherwise the section
        // would have inconsistent structure).
        let mut contracted_axis: Option<usize> = None;
        let mut contracted_ok = true;
        for &t_id in &terminators {
            let t_node = &model.nodes[t_id];
            let Ok(ax) = detect_contracted_score_axis(model, t_node, stream_sym) else {
                contracted_ok = false;
                break;
            };
            if let Some(prev) = contracted_axis
                && prev != ax
            {
                contracted_ok = false;
                break;
            }
            contracted_axis = Some(ax);
        }
        let Some(contracted_axis) = (if contracted_ok { contracted_axis } else { None }) else {
            continue;
        };

        sections.push(QuadraticSection { section, initiators, terminators, mask, contracted_axis });
    }

    Ok(sections)
}

/// Find the score-matrix axis (one of the two streaming axes of the
/// terminator's input 0) that is contracted away by the terminator op.
///
/// * Reduce<Sum>: it's the reduced axis (must be one of the streaming axes).
/// * EinSum: it's the streaming axis of input 0 that doesn't track to a
///   unique output axis (i.e. is summed over).
fn detect_contracted_score_axis(
    model: &TypedModel,
    terminator: &TypedNode,
    stream_sym: &Symbol,
) -> TractResult<usize> {
    let input_fact = model.outlet_fact(terminator.inputs[0])?;
    let streaming_axes = streaming_positions(input_fact, stream_sym);
    ensure!(
        streaming_axes.len() == 2,
        "Terminator score input has {} streaming axes, expected 2",
        streaming_axes.len()
    );
    if let Some(reduce) = terminator.op_as::<Reduce>() {
        for &ax in &streaming_axes {
            if reduce.axes.contains(&ax) {
                return Ok(ax);
            }
        }
        bail!("Reduce terminator doesn't reduce a streaming axis of the score input");
    }
    if let Some(einsum) = terminator.op_as::<EinSum>() {
        for &ax in &streaming_axes {
            let mapped = einsum.axes.track_axis((InOut::In(0), ax), InOut::Out(0))?;
            if mapped.is_none() {
                return Ok(ax);
            }
        }
        bail!("EinSum terminator doesn't contract any streaming axis of input 0");
    }
    bail!("Unsupported terminator op for contracted-axis detection: {}", terminator.op.name())
}

// Pattern is gone — see `rewrite` below, which derives the same per-op
// data from the QuadraticSection on the fly.

/// Recognise a mask `uniform_tdim` expression.  Returns the closed-enum
/// `MaskForm` description on success, `None` otherwise.
///
/// Today's recogniser handles two AST shapes, both reducing to the same
/// banded structure `chunk(axis_a) - chunk(axis_b) ∈ [lower, upper]`:
///
/// 1. `Eq(coord_a/k, coord_b/k)`              — block-diagonal (lower=upper=0)
/// 2. `Mul([Ge(upper, D), Ge(D, lower)])`     — banded, `D = coord_a/k - coord_b/k`
///
/// Both forms are produced by `core` after `reduce()` (see comparison.rs and
/// the And-of-Ge propagation in binary.rs).  Other AST shapes are rejected.
fn decode_mask(expr: &TDim, streaming_axes: &[usize]) -> Option<MaskForm> {
    if streaming_axes.len() != 2 {
        return None;
    }
    let want: BTreeSet<usize> = streaming_axes.iter().copied().collect();

    // Form 1 — block-diagonal Eq.
    if let TDim::Eq(lhs, rhs) = expr {
        let (axis_a, k_a) = decode_coord_div(lhs)?;
        let (axis_b, k_b) = decode_coord_div(rhs)?;
        if k_a != k_b {
            return None;
        }
        let got: BTreeSet<usize> = [axis_a, axis_b].into_iter().collect();
        if want != got {
            return None;
        }
        return Some(MaskForm { chunk_size: k_a as i64, lower: 0, upper: 0, axis_a, axis_b });
    }

    // Form 2 — banded Mul of two Ge's.
    if let TDim::Mul(terms) = expr
        && terms.len() == 2
    {
        for (a, b) in [(&terms[0], &terms[1]), (&terms[1], &terms[0])] {
            if let Some(form) = decode_banded_terms(a, b)
                && want == [form.axis_a, form.axis_b].into_iter().collect()
            {
                return Some(form);
            }
        }
    }
    None
}

/// `upper_term = Ge(Val(upper), D)` and `lower_term = Ge(D, Val(lower))`.
/// `D = coord_a/k - coord_b/k`.
fn decode_banded_terms(upper_term: &TDim, lower_term: &TDim) -> Option<MaskForm> {
    let TDim::Ge(u_val, d_upper) = upper_term else {
        return None;
    };
    let TDim::Val(upper) = **u_val else {
        return None;
    };
    let TDim::Ge(d_lower, l_val) = lower_term else {
        return None;
    };
    let TDim::Val(lower) = **l_val else {
        return None;
    };
    if d_lower != d_upper {
        return None;
    }
    let (axis_a, axis_b, k) = decode_diff(d_lower)?;
    Some(MaskForm { chunk_size: k as i64, lower, upper, axis_a, axis_b })
}

/// Match `Add([MulInt(-1, Div(Sym(🎯b), k)), Div(Sym(🎯a), k)])` (the canonical
/// `coord_a/k - coord_b/k` after `reduce()`) and return `(axis_a, axis_b, k)`.
fn decode_diff(expr: &TDim) -> Option<(usize, usize, u64)> {
    let TDim::Add(terms) = expr else {
        return None;
    };
    if terms.len() != 2 {
        return None;
    }
    for (pos, neg) in [(&terms[0], &terms[1]), (&terms[1], &terms[0])] {
        let Some((axis_a, k_a)) = decode_coord_div(pos) else {
            continue;
        };
        let TDim::MulInt(-1, neg_inner) = neg else {
            continue;
        };
        let Some((axis_b, k_b)) = decode_coord_div(neg_inner) else {
            continue;
        };
        if k_a == k_b {
            return Some((axis_a, axis_b, k_a));
        }
    }
    None
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
/// the chunked equivalent into the patch.  Unhandled op-types bubble up as
/// `Err` from the per-role dispatcher: a recognised section either gets
/// fully rewritten or fails loudly (no partial rewrites silently left for
/// downstream pulsification to trip over).
fn build_section_patch(
    model: &TypedModel,
    sec: &QuadraticSection,
    chunk_sym: &Symbol,
    k: i64,
) -> TractResult<TypedModelPatch> {
    ensure!(sec.mask.lower <= 0);
    ensure!(sec.mask.upper >= 0);
    let mut patch = TypedModelPatch::default();
    // Map from original outlet to its chunked equivalent inside the patch.
    let mut chunked: HashMap<OutletId, OutletId> = HashMap::default();
    // Boundary outlets to redirect via `shunt_outside` after wiring the
    // merge reshape: (original outlet, chunked-form outlet inside patch).
    let mut shunts: Vec<(OutletId, OutletId)> = vec![];

    // ── 1. Initiators ────────────────────────────────────────────────────
    // Mask-construction initiators (multi-T-axis with uniform_tdim) are
    // *skipped* — they're metadata the recogniser already consumed into
    // `MaskForm`, with no role in the chunked data flow.  At the body op
    // that consumes the mask wire, we'll substitute a constant block of
    // shape [1, k_a, k_b] (all-true/all-1 of the source dtype) — the
    // mask's value on the in-band region of every chunk.
    for &nid in &sec.initiators {
        let node = &model.nodes[nid];
        if node.outputs[0].fact.uniform_tdim.is_some() {
            continue;
        }
        let out =
            wire_initiator(&mut patch, model, node, &sec.mask, sec.contracted_axis, chunk_sym, k)?;
        chunked.insert(OutletId::new(nid, 0), out);
    }
    ensure!(!chunked.is_empty());

    // ── 2. Body ──────────────────────────────────────────────────────────
    // Walk the section in topological order, skipping initiators (already
    // wired), terminators (out-of-section by definition), and the
    // uniform_tdim mask-construction chain (handled at the body-op
    // boundary, see `wire_body`).
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
        let out = wire_body(
            &mut patch,
            model,
            node,
            &sec.mask,
            sec.contracted_axis,
            &chunked,
            chunk_sym,
            k,
        )?;
        chunked.insert(OutletId::new(nid, 0), out);
    }

    // ── 3. Terminators ───────────────────────────────────────────────────
    for &nid in &sec.terminators {
        let node = &model.nodes[nid];
        let (boundary, chunked_form) = wire_terminator(
            &mut patch,
            model,
            node,
            &chunked,
            &sec.mask,
            sec.contracted_axis,
            chunk_sym,
            k,
        )?;
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

    Ok(patch)
}

// ── Per-role dispatchers ────────────────────────────────────────────────
//
// Each `wire_*` helper takes a section node + the patch-in-progress and
// dispatches to a per-op-type implementation.  Unhandled op-types `bail!`
// with a clear "Unsupported …" message — Blockify either fully rewrites
// a detected section or errors loudly, never silently leaves a half-
// rewritten graph for downstream pulsification to choke on.

fn wire_initiator(
    patch: &mut TypedModelPatch,
    model: &TypedModel,
    node: &TypedNode,
    mask: &MaskForm,
    contracted_axis: usize,
    chunk_sym: &Symbol,
    k: i64,
) -> TractResult<OutletId> {
    if let Some(op) = node.op_as::<EinSum>() {
        return wire_initiator_einsum(patch, model, node, op, mask, contracted_axis, chunk_sym, k);
    }
    bail!("Unsupported initiator {node}")
}

/// Replay a body op in the chunked frame.
///
/// * Inputs from `chunked` (= data wires produced by the initiator path):
///   passed through as-is.
/// * Inputs whose source is a `uniform_tdim` wire (= the mask wire at
///   the boundary between the metadata chain and the data computation):
///   substituted with a `[1, k_a, k_b]` constant block — the mask's
///   value on the in-band region of every chunk.  For the recogniser's
///   current scope every position in the chunked block is in-band, so
///   the const is uniformly `true`/`1`/`1.0` of the source dtype.  No
///   per-op "mask role" knowledge needed; the body op sees a regular
///   tensor input it broadcasts over the chunk axis just like any const.
/// * Other external inputs: tapped, with `AddAxis(0)` bumping any
///   rank deficit so rank-strict ops (TypedBinOp, …) accept them.
///
/// `axes_mapping::track_axis` asserts each chunked input's chunk axis
/// reaches a unique output axis — bails with a precise error if the op
/// would disconnect the chunk axis (e.g. softmax over it).
fn wire_body(
    patch: &mut TypedModelPatch,
    model: &TypedModel,
    node: &TypedNode,
    mask: &MaskForm,
    contracted_axis: usize,
    chunked: &HashMap<OutletId, OutletId>,
    chunk_sym: &Symbol,
    k: i64,
) -> TractResult<OutletId> {
    // Pass 1: collect chunked inputs and discover the rank we'll work in.
    let n = node.inputs.len();
    let mut new_inputs: TVec<Option<OutletId>> = tvec![None; n];
    let mut chunk_input_axes: Vec<(usize, usize)> = vec![];
    let mut chunked_rank: Option<usize> = None;
    for (slot, &input) in node.inputs.iter().enumerate() {
        if let Some(&c) = chunked.get(&input) {
            let cf = patch.outlet_fact(c)?;
            let positions = streaming_positions(cf, chunk_sym);
            ensure!(
                positions.len() == 1,
                "Body op {node}: chunked input slot {slot} has {} streaming axes, expected 1",
                positions.len()
            );
            chunk_input_axes.push((slot, positions[0]));
            chunked_rank = Some(cf.rank());
            new_inputs[slot] = Some(c);
        }
    }
    let chunked_rank = chunked_rank.ok_or_else(|| {
        format_err!("Body op {node} has no chunked input — at least one is required")
    })?;

    // Pass 2: mask-wire substitutions (block const) and external taps
    // (rank-bumped to match the chunked frame).
    for (slot, &input) in node.inputs.iter().enumerate() {
        if new_inputs[slot].is_some() {
            continue;
        }
        let fact = model.outlet_fact(input)?;
        let in_outlet = if fact.uniform_tdim.is_some() {
            block_mask_const(patch, &node.name, slot, fact.datum_type, mask, contracted_axis, k)?
        } else {
            let tapped = patch.tap_model(model, input)?;
            bump_rank_to(patch, &node.name, slot, tapped, chunked_rank)?
        };
        new_inputs[slot] = Some(in_outlet);
    }
    let new_inputs: TVec<OutletId> = new_inputs.into_iter().map(|o| o.unwrap()).collect();

    // Chunkability: for every chunked input, its chunk axis must track
    // through to a unique output axis position.
    let input_facts: TVec<TypedFact> = new_inputs
        .iter()
        .map(|o| patch.outlet_fact(*o).map(|f| f.clone()))
        .collect::<TractResult<_>>()?;
    let in_refs: TVec<&TypedFact> = input_facts.iter().collect();
    let output_facts = node.op.output_facts(&in_refs)?;
    let out_refs: TVec<&TypedFact> = output_facts.iter().collect();
    let am = node.op.axes_mapping(&in_refs, &out_refs)?;
    for (slot, axis) in chunk_input_axes {
        let tracked = am.track_axis((InOut::In(slot), axis), InOut::Out(0))?;
        ensure!(
            tracked.is_some(),
            "Body op {node} doesn't preserve the chunk axis (input slot {slot}, axis {axis}) \
             through to the output — its axes_mapping disconnects it"
        );
    }

    Ok(patch.wire_node(&*node.name, node.op.clone(), &new_inputs)?[0])
}

/// `[1, k_a, k_b]` constant of the source mask wire's in-band value
/// (`true` for bool, `1`/`1.0` for numeric).  `k_a`/`k_b` are widened to
/// `W·k` on whichever side is the contracted (= windowed) axis.
fn block_mask_const(
    patch: &mut TypedModelPatch,
    node_name: &str,
    slot: usize,
    dt: DatumType,
    mask: &MaskForm,
    contracted_axis: usize,
    k: i64,
) -> TractResult<OutletId> {
    let window: i64 = mask.upper - mask.lower + 1;
    let k_a = if contracted_axis == mask.axis_a { window * k } else { k };
    let k_b = if contracted_axis == mask.axis_b { window * k } else { k };
    // Preserve the original axis order on the mask wire (axis_a < axis_b
    // means within-axis_a comes before within-axis_b in the block).
    let (within_first, within_second) =
        if mask.axis_a < mask.axis_b { (k_a, k_b) } else { (k_b, k_a) };
    let const_shape = [1usize, within_first as usize, within_second as usize];

    let const_tensor = if dt == bool::datum_type() {
        let count: usize = const_shape.iter().product();
        Tensor::from_shape(&const_shape, &vec![true; count])?
    } else {
        let scalar = tensor0(1i64).cast_to_dt(dt)?.into_owned();
        scalar.broadcast_to_shape(&const_shape)?
    };
    patch.add_const(format!("{node_name}.block_mask.{slot}"), const_tensor)
}

/// Insert `AddAxis(0)`s until the wire's rank reaches `target`.  Used to
/// rank-bump tapped external constants (e.g. a rank-2 broadcast literal
/// like the `0` in `ge(diff, 0)`) so they match the chunked-frame rank
/// for rank-strict consumer ops.
fn bump_rank_to(
    patch: &mut TypedModelPatch,
    node_name: &str,
    slot: usize,
    mut outlet: OutletId,
    target: usize,
) -> TractResult<OutletId> {
    let mut rank = patch.outlet_fact(outlet)?.rank();
    let mut step = 0;
    while rank < target {
        outlet = patch.wire_node(
            format!("{node_name}.bump_rank.{slot}.{step}"),
            AxisOp::Add(0),
            &[outlet],
        )?[0];
        rank += 1;
        step += 1;
    }
    Ok(outlet)
}

fn wire_terminator(
    patch: &mut TypedModelPatch,
    model: &TypedModel,
    node: &TypedNode,
    chunked: &HashMap<OutletId, OutletId>,
    mask: &MaskForm,
    contracted_axis: usize,
    chunk_sym: &Symbol,
    k: i64,
) -> TractResult<(OutletId, OutletId)> {
    if let Some(op) = node.op_as::<Reduce>() {
        return wire_terminator_reduce(patch, model, node, op, chunked);
    }
    if let Some(op) = node.op_as::<EinSum>() {
        return wire_terminator_einsum(
            patch,
            model,
            node,
            op,
            chunked,
            mask,
            contracted_axis,
            chunk_sym,
            k,
        );
    }
    bail!("Unsupported operator {node}")
}

// ── Per-op-type implementations ─────────────────────────────────────────

/// Initiator EinSum: tap each input from the model, wire a split reshape
/// for it, then wire the chunked EinSum.  For banded masks, additionally
/// wrap the input whose streaming axis tracks to `contracted_axis` (the
/// score-matrix axis the section's terminator contracts) with a
/// `WindowOnAxis(W)` + flatten reshape, so the within-chunk contracted
/// axis on that input has size `W·k` instead of `k`.  Returns the chunked
/// output.
fn wire_initiator_einsum(
    patch: &mut TypedModelPatch,
    model: &TypedModel,
    node: &TypedNode,
    op: &EinSum,
    mask: &MaskForm,
    contracted_axis: usize,
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

        // Banded path: if this input's stream axis is on the contracted
        // side of the section, expose `W` chunks per pulse on it.
        let tracked_in_score =
            op.axes.track_axis((InOut::In(ix), stream_axis), InOut::Out(0))?.ok_or_else(|| {
                format_err!(
                    "EinSum stream axis on input {ix} doesn't track to a unique output axis"
                )
            })?;
        let chunked = wrap_with_window_if_needed(
            patch,
            chunked,
            stream_axis,
            tracked_in_score,
            &format!("{}.{ix}", node.name),
            mask,
            contracted_axis,
            k,
        )?;
        chunked_inputs.push(chunked);
    }

    let in_starts: Vec<Option<usize>> = in_streaming_axes.iter().map(|&p| Some(p)).collect();
    let chunked_op = chunkify_einsum(op, &in_starts, Some(out_streaming_axes[0]))?;
    Ok(patch.wire_node(format!("{}.blockified", node.name), chunked_op, &chunked_inputs)?[0])
}

/// Wrap `chunked` (shape `[..., S, k, ...]` with the streaming dim at
/// `stream_axis`) with `WindowOnAxis(W) + flatten(W, k) → W·k` if the
/// section requires it: the mask is banded AND the input's stream axis
/// maps to the contracted score axis.  Otherwise pass through unchanged.
///
/// `score_axis` is where this input's stream axis lands on the score
/// matrix (= input 0 of the terminator).  `window_start_for(mask,
/// contracted_axis)` picks the slot offset so the W chunks cover the
/// in-band range relative to the consumer's logical chunk index.
fn wrap_with_window_if_needed(
    patch: &mut TypedModelPatch,
    chunked: OutletId,
    stream_axis: usize,
    score_axis: usize,
    name_prefix: &str,
    mask: &MaskForm,
    contracted_axis: usize,
    k: i64,
) -> TractResult<OutletId> {
    if mask.is_block_diag() || score_axis != contracted_axis {
        return Ok(chunked);
    }
    let window: usize = (mask.upper - mask.lower + 1) as usize;
    let start = window_start_for(mask, contracted_axis);
    let windowed = patch.wire_node(
        format!("{name_prefix}.window"),
        tract_pulse_opl::ops::WindowOnAxis { axis: stream_axis, window, start },
        &[chunked],
    )?[0];
    let from = tvec!(window.to_dim(), k.to_dim());
    let to = tvec!(((window as i64) * k).to_dim());
    let flatten = AxisOp::Reshape(stream_axis + 1, from, to);
    Ok(patch.wire_node(format!("{name_prefix}.window_flat"), flatten, &[windowed])?[0])
}

/// Slot-0 offset for a window that covers the in-band range:
///
/// * `contracted_axis == mask.axis_a`: at consumer logical chunk c on
///   the kept axis (= axis_b), we want `chunk(axis_a) ∈ [c + lower,
///   c + upper]` → slot 0 is at `c + lower`, so `start = lower`.
/// * `contracted_axis == mask.axis_b`: at consumer logical chunk c on
///   the kept axis (= axis_a), we want `chunk(axis_b) ∈ [c - upper,
///   c - lower]` → slot 0 is at `c - upper`, so `start = -upper`.
fn window_start_for(mask: &MaskForm, contracted_axis: usize) -> i64 {
    if contracted_axis == mask.axis_a { mask.lower } else { -mask.upper }
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
) -> TractResult<(OutletId, OutletId)> {
    ensure!(op.reducer == Reducer::Sum && op.axes.len() == 1);
    let chunked_input = chunked[&node.inputs[0]];
    // Chunk insertion position: the first streaming axis of the input fact.
    let in_fact = model.outlet_fact(node.inputs[0])?;
    let stream_sym = first_streaming_symbol(in_fact)?;
    let in_streaming = streaming_positions(in_fact, &stream_sym);
    ensure!(!in_streaming.is_empty());
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
            return Ok((OutletId::new(consumer.id, 0), chunked_rm));
        }
    }
    Ok((OutletId::new(node.id, 0), chunked_term))
}

/// EinSum terminator (e.g. ex02's `attn @ V`): chunkifies the second
/// EinSum the same way as the initiator.  Inputs already in `chunked`
/// (the multi-T-axis input from the body) are reused as-is; auxiliary
/// inputs (single-T-axis) get a tap + split reshape inserted.  For
/// banded masks, an auxiliary input whose stream axis maps (through
/// this einsum) to the section's `contracted_axis` of the score matrix
/// (= input 0 here) also gets `WindowOnAxis + flatten` so its
/// within-chunk axis matches the W·k size of the windowed score.
fn wire_terminator_einsum(
    patch: &mut TypedModelPatch,
    model: &TypedModel,
    node: &TypedNode,
    op: &EinSum,
    chunked: &HashMap<OutletId, OutletId>,
    mask: &MaskForm,
    contracted_axis: usize,
    chunk_sym: &Symbol,
    k: i64,
) -> TractResult<(OutletId, OutletId)> {
    let mut chunked_inputs: TVec<OutletId> = tvec!();
    let mut input_starts: Vec<Option<usize>> = vec![];
    for (slot, &input) in node.inputs.iter().enumerate() {
        let positions = streaming_positions(model.outlet_fact(input)?, chunk_sym);
        if let Some(&already_chunked) = chunked.get(&input) {
            // Multi-T-axis input from the body — already in chunked form
            // (windowed if needed by the initiator).
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

            // Where does this auxiliary's stream axis sit on the score
            // matrix (= input 0 of this einsum)?  If it's the contracted
            // side, window it so its within-chunk axis matches the
            // already-windowed score input.
            let aux_in_score = op.axes.track_axis((InOut::In(slot), stream_axis), InOut::In(0))?;
            let new_chunked = if let Some(score_axis) = aux_in_score {
                wrap_with_window_if_needed(
                    patch,
                    new_chunked,
                    stream_axis,
                    score_axis,
                    &format!("{}.in{slot}", node.name),
                    mask,
                    contracted_axis,
                    k,
                )?
            } else {
                new_chunked
            };
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
    Ok((OutletId::new(node.id, 0), chunked_term))
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
fn first_streaming_symbol(fact: &TypedFact) -> TractResult<Symbol> {
    fact.shape
        .iter()
        .find_map(|d| d.symbols().into_iter().next())
        .context("No streaming axis found")
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

    fn make_banded(scope: &SymbolScope, a: usize, b: usize, k: u64, lo: i64, up: i64) -> TDim {
        // Build the canonical form: Mul([Ge(Val(up), D), Ge(D, Val(lo))]).
        let div_a = TDim::Div(Box::new(coord(scope, a)), k);
        let div_b = TDim::Div(Box::new(coord(scope, b)), k);
        let diff = (div_a - div_b).reduce();
        let ge_upper = TDim::Ge(Box::new(TDim::Val(up)), Box::new(diff.clone())).reduce();
        let ge_lower = TDim::Ge(Box::new(diff), Box::new(TDim::Val(lo))).reduce();
        TDim::Mul(vec![ge_upper, ge_lower]).reduce()
    }

    #[test]
    fn decode_mask_recognises_block_diag_canonical_form() {
        let scope = SymbolScope::default();
        let expr = make_block_diag(&scope, 0, 1, 2);
        let m = decode_mask(&expr, &[0, 1]).unwrap();
        assert_eq!((m.chunk_size, m.lower, m.upper), (2, 0, 0));
    }

    #[test]
    fn decode_mask_recognises_block_diag_arbitrary_chunk_size() {
        let scope = SymbolScope::default();
        let expr = make_block_diag(&scope, 0, 1, 137);
        let m = decode_mask(&expr, &[0, 1]).unwrap();
        assert_eq!(m.chunk_size, 137);
    }

    #[test]
    fn decode_mask_recognises_block_diag_swapped_axes() {
        let scope = SymbolScope::default();
        let expr = make_block_diag(&scope, 1, 0, 2);
        let m = decode_mask(&expr, &[0, 1]).unwrap();
        assert_eq!(m.chunk_size, 2);
    }

    #[test]
    fn decode_mask_recognises_banded_form() {
        // Mimics ex03: `0 ≤ chunk(0) - chunk(1) ≤ 1` with k=2.
        let scope = SymbolScope::default();
        let expr = make_banded(&scope, 0, 1, 2, 0, 1);
        let m = decode_mask(&expr, &[0, 1]).unwrap();
        assert_eq!((m.chunk_size, m.lower, m.upper, m.axis_a, m.axis_b), (2, 0, 1, 0, 1));
    }

    #[test]
    fn decode_mask_recognises_banded_form_negative_lower() {
        let scope = SymbolScope::default();
        let expr = make_banded(&scope, 0, 1, 2, -1, 1);
        let m = decode_mask(&expr, &[0, 1]).unwrap();
        assert_eq!((m.chunk_size, m.lower, m.upper), (2, -1, 1));
    }

    #[test]
    fn decode_mask_rejects_mismatched_chunk_sizes() {
        let scope = SymbolScope::default();
        let expr = TDim::Eq(
            Box::new(TDim::Div(Box::new(coord(&scope, 0)), 2)),
            Box::new(TDim::Div(Box::new(coord(&scope, 1)), 3)),
        );
        assert_eq!(decode_mask(&expr, &[0, 1]), None);
    }

    #[test]
    fn decode_mask_rejects_non_streaming_axis() {
        let scope = SymbolScope::default();
        let expr = make_block_diag(&scope, 0, 2, 2);
        assert_eq!(decode_mask(&expr, &[0, 1]), None);
    }

    #[test]
    fn decode_mask_rejects_bare_ge() {
        let scope = SymbolScope::default();
        // A single Ge isn't a complete band — both bounds must be present.
        let expr = TDim::Ge(
            Box::new(TDim::Div(Box::new(coord(&scope, 0)), 2)),
            Box::new(TDim::Div(Box::new(coord(&scope, 1)), 2)),
        );
        assert_eq!(decode_mask(&expr, &[0, 1]), None);
    }

    /// Exploratory probe: confirm what the `(0 <= diff <= L) ∧ (mask)` form looks
    /// like at the TDim level after `reduce()`.  Kept as a regression on the
    /// canonical form the recogniser expects.
    #[test]
    fn decode_banded_probe_canonical_form() {
        let scope = SymbolScope::default();
        let coord_a = coord(&scope, 0);
        let coord_b = coord(&scope, 1);
        let div_a = TDim::Div(Box::new(coord_a), 2);
        let div_b = TDim::Div(Box::new(coord_b), 2);
        let diff = (div_a.clone() - div_b.clone()).reduce();
        let ge_lower = TDim::Ge(Box::new(diff.clone()), Box::new(TDim::Val(0))).reduce();
        let ge_upper = TDim::Ge(Box::new(TDim::Val(1)), Box::new(diff.clone())).reduce();
        let mask = TDim::Mul(vec![ge_upper, ge_lower]).reduce();
        println!("PROBE diff = {diff:?}");
        println!("PROBE mask = {mask:?}");
        println!("PROBE mask display = {mask}");
    }

    #[test]
    fn decode_mask_rejects_offset_in_numerator() {
        let scope = SymbolScope::default();
        let expr = TDim::Eq(
            Box::new(TDim::Div(Box::new(TDim::Add(vec![coord(&scope, 0), TDim::Val(1)])), 2)),
            Box::new(TDim::Div(Box::new(TDim::Add(vec![coord(&scope, 1), TDim::Val(1)])), 2)),
        );
        assert_eq!(decode_mask(&expr, &[0, 1]), None);
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
