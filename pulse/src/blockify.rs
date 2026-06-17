//! Blockify — typed-model rewrite that factors block-diagonal / banded
//! attention structure into the graph topology, so the result has a
//! single streaming axis everywhere and pulsifies under v1's existing
//! machinery.
//!
//! # Recogniser scope
//!
//! Banded masks `chunk(axis_a) − chunk(axis_b) ∈ [lower, upper]`
//! (block-diagonal is the special case `lower == upper == 0`):
//!
//!   EinSum([a, b]) producing a multi-T-axis score
//!   → body op(s) consuming scores and a mask wire whose
//!     `uniform_tdim` carries one of the recognised AST shapes
//!     (`Eq(coord_a/k, coord_b/k)` for block-diagonal, or
//!     `Mul([Ge(upper, D), Ge(D, lower)])` with `D = coord_a/k −
//!     coord_b/k` for banded — both forms are produced by core
//!     `reduce()` after Eq/And/Ge propagation)
//!   → Reduce<Sum> on a streaming axis, contracting EinSum, or
//!     ScaledMaskedSoftmax + downstream contracting EinSum
//!
//! `mask.lower > 0` (purely-future, skipping current) and `mask.upper < 0`
//! (purely-past) are rejected — they don't appear in attention masks and
//! would need different pulsifier wiring.
//!
//! # Pipeline
//!
//! 1. **Detect** quadratic sections globally (`find_quadratic_sections`):
//!    connected components of multi-T-axis nodes, with at least one
//!    `uniform_tdim`-annotated wire whose AST decodes to a `MaskForm`.
//!    Determine the score axis the terminator contracts and translate
//!    it to mask frame.
//! 2. **Substitute** the streaming symbol globally `T → k·S` via core's
//!    `set_symbols`.
//! 3. **Rewrite** one `TypedModelPatch` per section
//!    (`build_section_patch`).  Sections are independent so patches
//!    apply in sequence.  A recognised section gets fully rewritten or
//!    Blockify bails — no partial rewrites silently left for downstream
//!    pulsification to choke on.
//!
//! # Section rewrite
//!
//! Three initiator flavours (`wire_initiator` dispatches by op type):
//!
//! * **Data EinSum** (`wire_initiator_einsum`): the score-matrix
//!   producer.  Tap each input, split its T-axis at `k`, and on the
//!   contracted side wrap with `WindowOnAxis(W) + flatten(W, k) → W·k`
//!   so the chunked einsum's contracted within-chunk axis carries `W·k`
//!   rather than `k` elements.
//! * **uniform_tdim mask head** (`wire_uniform_tdim_initiator`): the
//!   multi-T-axis Sub/Eq at the top of the mask-construction chain.
//!   Each input is a single-T-axis chunk-id wire (`chunk_row`,
//!   `chunk_col`); `chunkify_uniform_tdim_input` taps it, casts TDim →
//!   I64 (PulsePad's `dispatch_copy_by_size!` fill needs `Copy`), splits
//!   the T-axis, moves the chunk axis to position 0, and on the
//!   contracted side wraps with `WindowOnAxis` using a **sentinel pad
//!   value** so out-of-stream boundary slots produce values way outside
//!   the band; downstream Ge/Le evaluate to `false` there.  After Sub,
//!   the result is cast back to the source dtype so downstream body ops
//!   that tap external constants (e.g. the `0` in `ge(diff, 0)`) match.
//! * **MultiBroadcastTo** (`wire_initiator_multibroadcastto`): the
//!   `select(mask, scores, scores * 0.0 + -inf)` false-branch pattern,
//!   where declutter folds the chain to a `MultiBroadcastTo` of a small
//!   const up to score's `[T, T]` shape.  The op's input is non-
//!   streaming, so we tap and rank-bump it to the chunked-frame rank;
//!   subsequent body-op broadcasting fills in the chunked dims with the
//!   constant value.
//!
//! Body ops (`wire_body`) are replayed op-cloned in the chunked frame.
//! Each input is one of:
//!
//! * **chunked** (in the `chunked` map): pass through.  May or may not
//!   carry a streaming axis — broadcast constants from the
//!   MultiBroadcastTo initiator have rank-bumped shape with no streaming
//!   axis, and that's fine.
//! * **other external** (taps): rank-bumped with `AddAxis(0)` to match
//!   the chunked frame so rank-strict consumers (TypedBinOp, …) accept
//!   them.
//!
//! `axes_mapping::track_axis` asserts each chunked input's chunk axis
//! reaches a unique output axis position — bails if the op would
//! disconnect it.  Body ops with explicit axis params (currently
//! `Softmax`) get axis indices shifted by `+1` via
//! `translate_body_op_axes` to account for the chunk axis inserted at
//! position 0.
//!
//! # Window-slot offsets
//!
//! * `contracted_axis == mask.axis_a`: `start = mask.lower` — consumer
//!   logical chunk `c` on the kept axis (= axis_b), window covers
//!   `chunk(axis_a) ∈ [c + lower, c + upper]`.
//! * `contracted_axis == mask.axis_b`: `start = -mask.upper` — kept
//!   axis = axis_a, window covers `chunk(axis_b) ∈ [c - upper, c -
//!   lower]`.
//!
//! `contracted_axis` lives in mask frame (0 or 1).  Score and mask align
//! via right-aligned broadcasting; the recogniser translates score-
//! frame axes from `axes_mapping::track_axis` to mask frame via
//! `axis - (score_rank - 2)`.
//!
//! Output `stream.delay` = `max(0, end_of_window)` chunks (positive when
//! the window extends past `c`, zero when fully causal).
//!
//! For EinSum terminators (e.g. attention's `attn @ V`), auxiliary
//! inputs whose stream axis tracks through the terminator einsum to the
//! contracted score axis are also windowed, so all inputs to the
//! terminator share the same W·k contracted-axis size.
//!
//! # Runtime dependencies
//!
//! * `tract_pulse_opl::ops::PulsedRange` — pulsifies the source's
//!   `Range(0, T)` chunk-id chain that
//!   `chunkify_uniform_tdim_input` taps.  Without it, `Range` falls
//!   through `NonPulsingWrappingOp` and produces a fresh symbolic shape
//!   the rest of pulsification can't match.
//! * `WindowOnAxis::pad_value` — set per-input to either `zero` (data
//!   wires) or a sentinel (chunk-id wires), depending on the initiator.
//!
//! # Known workarounds
//!
//! * Sentinel pad value bounded by `i32::MAX/4`: tract's `i64 → TDim`
//!   tensor cast routes through `i32` (`data/src/tensor.rs:1250`), so
//!   larger sentinels truncate to small junk and the band predicate
//!   evaluates true on boundary slots.  REVISIT: fix the cast upstream
//!   and lift the cap.
//! * TDim → I64 cast on chunk-id wires before windowing, then back to
//!   TDim after the chunked Sub: `PulsePad`'s fill uses
//!   `dispatch_copy_by_size!` which doesn't include TDim (not `Copy`).
//!   REVISIT: add a clone-fill arm to `PulsePad` for TDim and drop the
//!   round-trip.

use crate::internal::*;
use std::collections::{BTreeSet, HashMap};
use tract_core::axes::AxesMapping;
use tract_core::model::TypedModelPatch;
use tract_core::ops::binary::TypedBinOp;
use tract_core::ops::change_axes::AxisOp;
use tract_core::ops::einsum::EinSum;
use tract_core::ops::nn::{Reduce, Reducer};
use tract_core::transform::ModelTransform;
use tract_transformers::ops::DiagGather;

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
                 sizes; a single global substitution cannot cover them.  \
                 Refusing to blockify rather than produce a partial rewrite."
            );
        }

        let chunk_sym = model.symbols.new_with_prefix("S");
        let subs: HashMap<Symbol, TDim> =
            HashMap::from([(stream_sym.clone(), chunk_sym.to_dim() * k)]);
        let new_model = model.set_symbols(&subs)?;
        *model = new_model;
        rewrite_sections(model, &chunk_sym, k)?;
        model.properties.insert(
            BLOCKIFY_ORIGINAL_SYMBOL.to_string(),
            tensor1(&[symbol_name.to_string()]).into_arc_tensor(),
        );
        Ok(())
    }
}

pub fn has_quadratic_sections(model: &TypedModel, stream_sym: &Symbol) -> TractResult<bool> {
    Ok(!find_quadratic_sections(model, stream_sym)?.is_empty())
}

/// Rewrite every quadratic section in `model`.  The model is expected to
/// already be in post-substitute form (streaming dim = `multiplier · chunk_sym`).
/// `substitute_multiplier` is recorded as `BLOCKIFY_CHUNK_SIZE` for downstream
/// pulse-value translation; the section rewrite itself uses each section's
/// own `mask.chunk_size` for its chunked Reshape.
pub fn rewrite_sections(
    model: &mut TypedModel,
    chunk_sym: &Symbol,
    substitute_multiplier: i64,
) -> TractResult<bool> {
    let sections = find_quadratic_sections(model, chunk_sym)?;
    if sections.is_empty() {
        return Ok(false);
    }
    let k = sections[0].mask.chunk_size;
    if !sections.iter().all(|s| s.mask.chunk_size == k) {
        bail!(
            "Blockify found multiple quadratic sections with mismatched chunk \
             sizes; a single global substitution cannot cover them.  \
             Refusing to blockify rather than produce a partial rewrite."
        );
    }

    for sec in &sections {
        let patch = build_section_patch(model, sec, chunk_sym, sec.mask.chunk_size)?;
        patch.apply(model)?;
    }

    model.properties.insert(
        BLOCKIFY_CHUNK_SYMBOL.to_string(),
        tensor1(&[format!("{chunk_sym}")]).into_arc_tensor(),
    );
    model
        .properties
        .insert(BLOCKIFY_CHUNK_SIZE.to_string(), tensor0(substitute_multiplier).into_arc_tensor());
    Ok(true)
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

/// If `einsum_node`'s only multi-T-axis successor in `sec` is a single
/// DiagGather, return its node id — the pair forms a fused initiator
/// (DiagGather drives the chunked rewrite; the einsum is tapped through).
fn section_only_diag_gather_consumer(
    model: &TypedModel,
    einsum_node: &TypedNode,
    sec: &QuadraticSection,
) -> Option<usize> {
    let consumers: Vec<_> = model
        .outlet_successors(OutletId::new(einsum_node.id, 0))
        .iter()
        .filter(|s| sec.section.contains(&s.node))
        .collect();
    if consumers.len() != 1 {
        return None;
    }
    let dg_id = consumers[0].node;
    if !model.nodes[dg_id].op_is::<DiagGather>() {
        return None;
    }
    Some(dg_id)
}

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
/// terminator's input 0) that is contracted away by the terminator op,
/// **translated into mask frame** (so it's directly comparable to
/// `mask.axis_a` / `mask.axis_b`).  Score and mask align via right-
/// aligned broadcasting; mask is always rank-2 in the recogniser scope,
/// so the translation is `axis - (score_rank - 2)`.
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
    let score_rank = input_fact.rank();
    let rank_diff = score_rank
        .checked_sub(2)
        .ok_or_else(|| format_err!("Terminator score input rank {score_rank} < 2; expected ≥ 2"))?;
    let to_mask_frame = |score_axis: usize| -> TractResult<usize> {
        score_axis.checked_sub(rank_diff).ok_or_else(|| {
            format_err!(
                "Terminator score axis {score_axis} doesn't map to mask frame \
                 (rank_diff={rank_diff})"
            )
        })
    };
    if let Some(reduce) = terminator.op_as::<Reduce>() {
        for &ax in &streaming_axes {
            if reduce.axes.contains(&ax) {
                return to_mask_frame(ax);
            }
        }
        bail!("Reduce terminator doesn't reduce a streaming axis of the score input");
    }
    if let Some(einsum) = terminator.op_as::<EinSum>() {
        for &ax in &streaming_axes {
            let mapped = einsum.axes.track_axis((InOut::In(0), ax), InOut::Out(0))?;
            if mapped.is_none() {
                return to_mask_frame(ax);
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
    // Nodes wired as part of a fused initiator (e.g. einsum → DiagGather
    // for the Transformer-XL relative-position pattern).  These should be
    // skipped by the regular initiator and body loops since their
    // chunked equivalent is already in `chunked`.
    let mut already_wired: BTreeSet<usize> = BTreeSet::new();
    // Boundary outlets to redirect via `shunt_outside` after wiring the
    // merge reshape: (original outlet, chunked-form outlet inside patch).
    let mut shunts: Vec<(OutletId, OutletId)> = vec![];

    // Fused EinSum + DiagGather initiator: when an EinSum's only
    // multi-T-axis section consumer is a DiagGather, route the pair
    // through DiagGather's chunker and mark both as already-wired.
    for &nid in &sec.initiators {
        let einsum_node = &model.nodes[nid];
        if !einsum_node.op_is::<EinSum>() {
            continue;
        }
        let Some(dg_id) = section_only_diag_gather_consumer(model, einsum_node, sec) else {
            continue;
        };
        let dg_node = &model.nodes[dg_id];
        let dg_in_fact = model.outlet_fact(dg_node.inputs[0])?;
        let dg_in_streaming = streaming_positions(dg_in_fact, chunk_sym);
        if dg_in_streaming.len() != 1 {
            bail!(
                "EinSum+DiagGather initiator: DG input must have a single streaming axis, got {dg_in_streaming:?}"
            );
        }
        let dg_op = dg_node.op_as::<DiagGather>().unwrap();
        let dg_chunked = wire_initiator_diag_gather(
            &mut patch,
            model,
            dg_node,
            dg_op,
            &sec.mask,
            sec.contracted_axis,
            chunk_sym,
            k,
        )?;
        chunked.insert(OutletId::new(nid, 0), dg_chunked);
        chunked.insert(OutletId::new(dg_id, 0), dg_chunked);
        already_wired.insert(nid);
        already_wired.insert(dg_id);
    }

    // ── 1. Initiators ────────────────────────────────────────────────────
    // Two flavours:
    //   - Data initiators (e.g. score-matrix EinSum): tap, split, optional
    //     WindowOnAxis on the contracted side, wire chunked op.
    //   - Mask-construction initiators (multi-T-axis with uniform_tdim,
    //     typically Eq/Sub of two single-T-axis chunk-index wires): tap
    //     each input, split its T-axis, move the chunk axis to position
    //     0, optionally WindowOnAxis on the contracted side with a
    //     **sentinel pad value** so the band predicate on out-of-stream
    //     boundary slots evaluates to false.  Wire the binop with the
    //     chunked inputs.  The result lives in `chunked` like any other
    //     section wire.
    for &nid in &sec.initiators {
        if already_wired.contains(&nid) {
            continue;
        }
        let node = &model.nodes[nid];
        let out = if node.outputs[0].fact.uniform_tdim.is_some() {
            wire_uniform_tdim_initiator(
                &mut patch,
                model,
                node,
                &sec.mask,
                sec.contracted_axis,
                chunk_sym,
                k,
            )?
        } else {
            wire_initiator(&mut patch, model, node, &sec.mask, sec.contracted_axis, chunk_sym, k)?
        };
        chunked.insert(OutletId::new(nid, 0), out);
    }
    ensure!(!chunked.is_empty());

    // ── 2. Body ──────────────────────────────────────────────────────────
    // Walk the section in topological order, skipping initiators (already
    // wired) and terminators (out-of-section by definition).  Multi-T-axis
    // uniform_tdim body ops (e.g. Ge/Le/And/Cast on the chunked mask
    // chain) are processed like any other body op now: their inputs come
    // from `chunked` (the upstream chunked mask outlet), their outputs
    // feed back into `chunked` for downstream consumers.
    for &nid in &model.eval_order()? {
        if !sec.section.contains(&nid) {
            continue;
        }
        if sec.initiators.contains(&nid) {
            continue;
        }
        if already_wired.contains(&nid) {
            continue;
        }
        let node = &model.nodes[nid];
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
        let merged = wire_affine_tail_pad(&mut patch, model, boundary, merged, chunk_sym, k)?;
        patch.shunt_outside(model, boundary, merged)?;
    }

    Ok(patch)
}

/// Pad the chunked outlet with `c` constant-zero frames to match a
/// boundary outlet with streaming dim `c + k·S` (vs the merged `k·S`).
/// Restores the tail `wire_chunk_split` trimmed pre-section.
fn wire_affine_tail_pad(
    patch: &mut TypedModelPatch,
    model: &TypedModel,
    boundary: OutletId,
    merged: OutletId,
    chunk_sym: &Symbol,
    k: i64,
) -> TractResult<OutletId> {
    let boundary_fact = model.outlet_fact(boundary)?;
    let merged_fact = patch.outlet_fact(merged)?.clone();
    if boundary_fact.shape.len() != merged_fact.shape.len() {
        return Ok(merged);
    }
    let mut pad_axis: Option<(usize, i64)> = None;
    for (axis, (b, m)) in boundary_fact.shape.iter().zip(merged_fact.shape.iter()).enumerate() {
        if b == m {
            continue;
        }
        let b_off = affine_chunk_offset(b, chunk_sym, k);
        let m_off = affine_chunk_offset(m, chunk_sym, k);
        match (b_off, m_off) {
            (Some(bc), Some(0)) if bc > 0 => {
                if pad_axis.is_some() {
                    return Ok(merged);
                }
                pad_axis = Some((axis, bc));
            }
            _ => return Ok(merged),
        }
    }
    let Some((axis, c)) = pad_axis else {
        return Ok(merged);
    };
    let mut pads = vec![(0usize, 0usize); merged_fact.shape.len()];
    pads[axis] = (0, c as usize);
    let pad_value = Tensor::zero_scalar_dt(merged_fact.datum_type)?.into_arc_tensor();
    let pad_op = tract_core::ops::array::Pad {
        pads,
        mode: tract_core::ops::array::PadMode::Constant(pad_value),
    };
    let name = format!("{}.affine_tail_pad", model.nodes[boundary.node].name);
    Ok(patch.wire_node(name, pad_op, &[merged])?[0])
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
    if node.op_as::<tract_core::ops::array::MultiBroadcastTo>().is_some() {
        let in_fact = model.outlet_fact(node.inputs[0])?;
        if streaming_positions(in_fact, chunk_sym).is_empty() {
            return wire_initiator_multibroadcastto(patch, model, node, chunk_sym);
        } else {
            return wire_initiator_multibroadcastto_streaming(
                patch,
                model,
                node,
                mask,
                contracted_axis,
                chunk_sym,
                k,
            );
        }
    }
    if let Some(op) = node.op_as::<DiagGather>() {
        return wire_initiator_diag_gather(
            patch,
            model,
            node,
            op,
            mask,
            contracted_axis,
            chunk_sym,
            k,
        );
    }
    if let Some(op) = node.op_as::<TypedBinOp>() {
        return wire_initiator_typed_binop(
            patch,
            model,
            node,
            op,
            mask,
            contracted_axis,
            chunk_sym,
            k,
        );
    }
    bail!("Unsupported initiator {node}")
}

/// Initiator for `DiagGather` — the folded skew trick at the section
/// boundary.  Input is single-T-axis (the relative-position pre-skew
/// scores `[..., S, 2T_max-1]`), output is multi-T-axis (`[..., S, S]`,
/// the absolute-position scores).  In the chunked frame both wires
/// become single-T-axis with constant inner shape:
///
///   input  [..., chunks, k, 2T_max-1]
///   output [..., chunks, k, W]    where W = (mask.upper-mask.lower+1)·k
///
/// The chunked DiagGather has fixed `offset = k-1` (= P-1, the relative-
/// position-zero entry within the per-pulse window) and `out_len = W`.
#[allow(clippy::too_many_arguments)]
fn wire_initiator_diag_gather(
    patch: &mut TypedModelPatch,
    model: &TypedModel,
    node: &TypedNode,
    op: &DiagGather,
    mask: &MaskForm,
    contracted_axis: usize,
    chunk_sym: &Symbol,
    k: i64,
) -> TractResult<OutletId> {
    let out_streaming = streaming_positions(&node.outputs[0].fact, chunk_sym);
    ensure!(
        out_streaming.len() == 2 && out_streaming[1] == out_streaming[0] + 1,
        "Initiator DiagGather output must have two contiguous streaming axes, \
         got {out_streaming:?}"
    );
    ensure!(node.inputs.len() == 1, "DiagGather has 1 input, got {}", node.inputs.len());

    let in_fact = model.outlet_fact(node.inputs[0])?;
    let in_streaming = streaming_positions(in_fact, chunk_sym);
    ensure!(
        in_streaming.len() == 1,
        "Initiator DiagGather input must have exactly one streaming axis, got {in_streaming:?}"
    );
    let stream_axis = in_streaming[0];

    let tapped = patch.tap_model(model, node.inputs[0])?;
    let in_fact_patch = patch.outlet_fact(tapped)?.clone();
    let chunked = wire_chunk_split(patch, &node.name, tapped, stream_axis, chunk_sym, k)?;

    // The R (relative-position) axis of pos_raw is the last axis: a constant
    // width carrying the rel-pos table.  The DiagGather op's `offset` field
    // points to the column where rel-pos = 0 lives in the R-axis numbering.
    // Per chunk c, the W = (L+1)·k key-window starts at chunk
    // `c + window_start` (= `c − L` for lookback, `c` for lookahead), so the
    // (δi, δj)-th in-window key has rel-pos `δj − δi + window_start·k`.
    // Solving `chunked_offset + δj − δi = (op.offset) + (δj − δi + window_start·k)`
    // gives `chunked_offset = op.offset + window_start·k`.
    let r_axis = in_fact_patch.shape.last().context("DiagGather input has no last axis")?;
    let r = r_axis.to_i64().context("DiagGather R axis must be a constant integer")?;
    // Prefer `op.offset` if it simplifies to a concrete column index — this is
    // the path the streaming-rel-pos rewrite (subsequent commit) uses to plant
    // a centre that doesn't match the canonical `(R-1)/2`.  Fall back to the
    // T-XL convention `centre = (R-1)/2` for models where the op was built
    // with a row-count-based symbolic offset (e.g. `T - 1`) that hasn't
    // simplified post-substitution.
    let centre = op.offset.to_i64().ok().unwrap_or((r - 1) / 2);
    let l = mask.upper - mask.lower;
    let w = (l + 1) * k;
    let window_start = window_start_for(mask, contracted_axis);
    let chunked_offset = centre + window_start * k;
    let chunked_op = DiagGather { offset: chunked_offset.to_dim(), out_len: w.to_dim() };
    Ok(patch.wire_node(format!("{}.blockified", node.name), chunked_op, &[chunked])?[0])
}

/// Generic initiator for a `TypedBinOp` lifting two single-T-axis inputs
/// into a multi-T-axis score-shape output via implicit broadcasting (post-
/// declutter spelling of the pad-mask outer-AND pattern: `Add(at=0)(pad)` AND
/// `Add(at=1)(pad)` → `[T, T]`).
///
/// Per input, the streaming axis tracks via the op's axes_mapping to one of
/// the section's score axes.  If that score axis lands on the contracted (K)
/// side of the mask, the input is windowed by `WindowOnAxis(W) + flatten`;
/// otherwise it's just chunk-split.  Each input's chunks axis is then moved
/// to the section's `chunks_target_axis` so the chunked op aligns them.
///
/// `WindowOnAxis` pads boundary slots with the op's `absorbing_element` (0
/// for And/Mul/BitAnd, 1 for Or), so the chunked op produces "definitely
/// excluded" at out-of-stream positions.  Bails if the op has no absorbing
/// element (e.g. Add, Xor) — we can't safely window-pad those.
///
/// Non-streaming inputs are tapped and rank-bumped to the chunked-frame rank
/// (= score_rank + 1).  The chunked op's own broadcasting fills in the
/// streaming dims.
#[allow(clippy::too_many_arguments)]
fn wire_initiator_typed_binop(
    patch: &mut TypedModelPatch,
    model: &TypedModel,
    node: &TypedNode,
    op: &TypedBinOp,
    mask: &MaskForm,
    contracted_axis: usize,
    chunk_sym: &Symbol,
    k: i64,
) -> TractResult<OutletId> {
    let out_streaming_axes = streaming_positions(&node.outputs[0].fact, chunk_sym);
    ensure!(
        out_streaming_axes.len() == 2 && out_streaming_axes[1] == out_streaming_axes[0] + 1,
        "Initiator TypedBinOp output must have two contiguous streaming axes"
    );
    let chunks_target_axis = out_streaming_axes[0];
    let score_rank = node.outputs[0].fact.rank();
    let rank_diff = score_rank.checked_sub(2).ok_or_else(|| {
        format_err!("Score rank {score_rank} < 2; cannot translate to mask frame")
    })?;

    let input_facts: TVec<&TypedFact> =
        node.inputs.iter().map(|inp| model.outlet_fact(*inp)).collect::<TractResult<_>>()?;
    let output_facts: TVec<&TypedFact> = node.outputs.iter().map(|o| &o.fact).collect();
    let mapping = op.axes_mapping(&input_facts, &output_facts)?;

    let mut chunked_inputs: TVec<OutletId> = tvec!();
    for (ix, &input) in node.inputs.iter().enumerate() {
        let in_fact = model.outlet_fact(input)?;
        let streaming = streaming_positions(in_fact, chunk_sym);
        ensure!(
            streaming.len() <= 1,
            "Initiator TypedBinOp input {ix} has {} streaming axes, expected 0 or 1",
            streaming.len()
        );

        let tapped = patch.tap_model(model, input)?;
        let wire = if streaming.is_empty() {
            let target_rank = score_rank + 1;
            bump_rank_to(patch, &node.name, ix, tapped, target_rank)?
        } else {
            let stream_axis = streaming[0];
            let split = wire_chunk_split(
                patch,
                &format!("{}.{ix}", node.name),
                tapped,
                stream_axis,
                chunk_sym,
                k,
            )?;

            let tracked_in_score = mapping
                .track_axis((InOut::In(ix), stream_axis), InOut::Out(0))?
                .ok_or_else(|| {
                    format_err!(
                        "TypedBinOp stream axis on input {ix} doesn't track to a unique output axis"
                    )
                })?;
            let tracked_in_mask = tracked_in_score.checked_sub(rank_diff).ok_or_else(|| {
                format_err!(
                    "Tracked score axis {tracked_in_score} doesn't map to mask frame \
                     (rank_diff={rank_diff})"
                )
            })?;

            let needs_window = !mask.is_block_diag() && tracked_in_mask == contracted_axis;
            let after_window = if needs_window {
                let window: usize = (mask.upper - mask.lower + 1) as usize;
                let start = window_start_for(mask, contracted_axis);
                let dt = patch.outlet_fact(split)?.datum_type;
                let absorbing = op.0.absorbing_element().ok_or_else(|| {
                    format_err!(
                        "TypedBinOp '{}' has no absorbing_element; cannot safely window-pad \
                         a section-initiator input",
                        op.0.name()
                    )
                })?;
                let pad_value = tensor0(absorbing).cast_to_dt(dt)?.into_owned().into_arc_tensor();
                let windowed = patch.wire_node(
                    format!("{}.{ix}.window", node.name),
                    tract_pulse_opl::ops::WindowOnAxis {
                        axis: stream_axis,
                        window,
                        start,
                        pad_value,
                    },
                    &[split],
                )?[0];
                let from = tvec!(window.to_dim(), k.to_dim());
                let to = tvec!(((window as i64) * k).to_dim());
                patch.wire_node(
                    format!("{}.{ix}.window_flat", node.name),
                    AxisOp::Reshape(stream_axis + 1, from, to),
                    &[windowed],
                )?[0]
            } else {
                split
            };

            if stream_axis != chunks_target_axis {
                patch.wire_node(
                    format!("{}.{ix}.move_chunks", node.name),
                    AxisOp::Move(stream_axis, chunks_target_axis),
                    &[after_window],
                )?[0]
            } else {
                after_window
            }
        };
        chunked_inputs.push(wire);
    }

    Ok(patch.wire_node(format!("{}.blockified", node.name), op.clone(), &chunked_inputs)?[0])
}

/// Initiator for `MultiBroadcastTo` — the `select(mask, scores, -inf)`
/// false-branch pattern, where declutter folds `scores * 0.0 + -inf`
/// down to a `MultiBroadcastTo` of a small const (typically scalar) up
/// to the score's `[T, T]` shape.  The op's input is non-streaming
/// (otherwise `MultiBroadcastTo` would be in the middle of the section,
/// not its boundary), so we just tap and rank-bump it to the chunked-
/// frame rank `score_rank + 1`.  Subsequent body-op broadcasting fills
/// in the chunked dimensions with the (constant) input value.
fn wire_initiator_multibroadcastto(
    patch: &mut TypedModelPatch,
    model: &TypedModel,
    node: &TypedNode,
    chunk_sym: &Symbol,
) -> TractResult<OutletId> {
    ensure!(node.inputs.len() == 1, "MultiBroadcastTo expects 1 input, got {}", node.inputs.len());
    let input = node.inputs[0];
    let in_fact = model.outlet_fact(input)?;
    ensure!(
        streaming_positions(in_fact, chunk_sym).is_empty(),
        "MultiBroadcastTo initiator with streaming input not supported (input has \
         {} streaming axes)",
        streaming_positions(in_fact, chunk_sym).len(),
    );
    let target_rank = node.outputs[0].fact.rank() + 1;
    let mut wire = patch.tap_model(model, input)?;
    let mut step = 0;
    while patch.outlet_fact(wire)?.rank() < target_rank {
        wire =
            patch.wire_node(format!("{}.bump_rank.{step}", node.name), AxisOp::Add(0), &[wire])?[0];
        step += 1;
    }
    Ok(wire)
}

/// Initiator for `MultiBroadcastTo` whose input is streaming:
/// a `[..., 1, T]` per-key-position mask broadcast to `[..., T, T]`.
/// The input's streaming axis must track (after broadcast) to the
/// section's `contracted_axis`.  Chunked form per chunk: split T into
/// `[S, k]`, window L+1 chunks, flatten `[L+1, k] → W`, move the chunk
/// axis to first-streaming position, broadcast the size-1 axis up to k.
fn wire_initiator_multibroadcastto_streaming(
    patch: &mut TypedModelPatch,
    model: &TypedModel,
    node: &TypedNode,
    mask: &MaskForm,
    contracted_axis: usize,
    chunk_sym: &Symbol,
    k: i64,
) -> TractResult<OutletId> {
    ensure!(node.inputs.len() == 1, "MultiBroadcastTo expects 1 input, got {}", node.inputs.len());
    let input = node.inputs[0];
    let in_fact = model.outlet_fact(input)?;
    let in_streaming = streaming_positions(in_fact, chunk_sym);
    ensure!(
        in_streaming.len() == 1,
        "MultiBroadcastTo streaming initiator: input must have exactly one streaming axis, \
         got {in_streaming:?}"
    );
    let in_stream_axis = in_streaming[0];

    let out_streaming = streaming_positions(&node.outputs[0].fact, chunk_sym);
    ensure!(
        out_streaming.len() == 2 && out_streaming[1] == out_streaming[0] + 1,
        "Initiator MultiBroadcastTo output must have two contiguous streaming axes, \
         got {out_streaming:?}"
    );

    // The input axis that gets broadcast from 1 to a streaming dim.
    let bcast_axis = if out_streaming[0] == in_stream_axis {
        out_streaming[1]
    } else if out_streaming[1] == in_stream_axis {
        out_streaming[0]
    } else {
        bail!(
            "MultiBroadcastTo streaming initiator: input stream axis {in_stream_axis} not in \
             output streaming axes {out_streaming:?}"
        );
    };
    ensure!(
        in_fact.shape[bcast_axis].is_one(),
        "MultiBroadcastTo streaming initiator: broadcast-from axis {bcast_axis} must be 1, \
         got {}",
        in_fact.shape[bcast_axis]
    );

    // Translate the score-frame stream axis to mask frame and check it's the
    // contracted side.  `score_rank - 2` is the leading-batch-dims offset.
    let score_rank = node.outputs[0].fact.rank();
    let rank_diff = score_rank.checked_sub(2).ok_or_else(|| {
        format_err!("Score rank {score_rank} < 2; cannot translate to mask frame")
    })?;
    let tracked_in_mask = in_stream_axis.checked_sub(rank_diff).ok_or_else(|| {
        format_err!(
            "Tracked score axis {in_stream_axis} doesn't map to mask frame (rank_diff={rank_diff})"
        )
    })?;
    ensure!(
        tracked_in_mask == contracted_axis,
        "MultiBroadcastTo streaming initiator: input stream axis must track to the \
         contracted axis ({contracted_axis}), got {tracked_in_mask}"
    );

    let tapped = patch.tap_model(model, input)?;
    let split = wire_chunk_split(patch, &node.name, tapped, in_stream_axis, chunk_sym, k)?;
    let bcast_axis_post_split =
        if bcast_axis > in_stream_axis { bcast_axis + 1 } else { bcast_axis };

    let window: usize = (mask.upper - mask.lower + 1) as usize;
    let start = window_start_for(mask, contracted_axis);
    let dt = patch.outlet_fact(split)?.datum_type;
    let pad_value = Tensor::zero_scalar_dt(dt)?.into_arc_tensor();
    let windowed = patch.wire_node(
        format!("{}.window", node.name),
        tract_pulse_opl::ops::WindowOnAxis { axis: in_stream_axis, window, start, pad_value },
        &[split],
    )?[0];
    let bcast_axis_post_window = if bcast_axis_post_split > in_stream_axis {
        bcast_axis_post_split + 1
    } else {
        bcast_axis_post_split
    };

    // Flatten [L+1, k] back to a single W = (L+1)·k axis.
    let from = tvec!(window.to_dim(), k.to_dim());
    let to = tvec!(((window as i64) * k).to_dim());
    let flat = patch.wire_node(
        format!("{}.window_flat", node.name),
        AxisOp::Reshape(in_stream_axis + 1, from, to),
        &[windowed],
    )?[0];
    let bcast_axis_post_flat = if bcast_axis_post_window > in_stream_axis + 1 {
        bcast_axis_post_window - 1
    } else {
        bcast_axis_post_window
    };

    // Move chunk axis to the original first-streaming output position
    // (convention shared with `chunkify_einsum`).
    let chunks_target_axis = out_streaming[0];
    let mut chunks_axis = in_stream_axis;
    let mut bcast_axis_now = bcast_axis_post_flat;
    let mut wire = flat;
    if chunks_axis != chunks_target_axis {
        wire = patch.wire_node(
            format!("{}.move_chunks", node.name),
            AxisOp::Move(chunks_axis, chunks_target_axis),
            &[wire],
        )?[0];
        // Track the broadcast-from-1 axis through the Move: dims STRICTLY
        // between source and target shift by one slot — [target, source)
        // leftward, (source, target] rightward.
        if chunks_target_axis < chunks_axis {
            if bcast_axis_now >= chunks_target_axis && bcast_axis_now < chunks_axis {
                bcast_axis_now += 1;
            }
        } else if bcast_axis_now > chunks_axis && bcast_axis_now <= chunks_target_axis {
            bcast_axis_now = bcast_axis_now.saturating_sub(1);
        }
        chunks_axis = chunks_target_axis;
        let _ = chunks_axis;
    }

    let mut target_shape: TVec<TDim> = patch.outlet_fact(wire)?.shape.to_tvec();
    target_shape[bcast_axis_now] = k.to_dim();
    let bcast = tract_core::ops::array::MultiBroadcastTo { shape: target_shape.into() };
    Ok(patch.wire_node(format!("{}.blockified", node.name), bcast, &[wire])?[0])
}

/// Initiator for a multi-T-axis `uniform_tdim` node — typically the
/// `Eq`/`Sub` head of the mask-construction chain whose two inputs are
/// single-T-axis chunk-index wires (`chunk_row` at axis 0, `chunk_col`
/// at axis 1).  Tap each input, split its T-axis into `[..., S, k, ...]`,
/// move the chunk axis to position 0 to align with the rest of the
/// section, and (if its source T-axis equals the section's contracted
/// axis) wrap with `WindowOnAxis` using a **sentinel pad value** so the
/// downstream band predicate evaluates to false on out-of-stream
/// boundary slots.  Then wire the same op (Eq/Sub/…) with the chunked
/// inputs.
fn wire_uniform_tdim_initiator(
    patch: &mut TypedModelPatch,
    model: &TypedModel,
    node: &TypedNode,
    mask: &MaskForm,
    contracted_axis: usize,
    chunk_sym: &Symbol,
    k: i64,
) -> TractResult<OutletId> {
    let mut chunked_inputs: TVec<OutletId> = tvec!();
    for (ix, &input) in node.inputs.iter().enumerate() {
        let chunked = chunkify_uniform_tdim_input(
            patch,
            model,
            input,
            &format!("{}.in{ix}", node.name),
            mask,
            contracted_axis,
            chunk_sym,
            k,
        )?;
        chunked_inputs.push(chunked);
    }
    let mut out =
        patch.wire_node(format!("{}.blockified", node.name), node.op.clone(), &chunked_inputs)?[0];
    // Match the source's output dtype: `chunkify_uniform_tdim_input` may
    // have cast TDim → I64 to satisfy `PulsePad`'s Copy-based fill, so
    // body ops downstream that tap external constants (e.g. the `0` in
    // `ge(diff, 0)`) need the chunked outlet to carry the original dtype.
    let source_dt = node.outputs[0].fact.datum_type;
    let cur_dt = patch.outlet_fact(out)?.datum_type;
    if cur_dt != source_dt {
        out = patch.wire_node(
            format!("{}.blockified.cast_back", node.name),
            tract_core::ops::cast::cast(source_dt),
            &[out],
        )?[0];
    }
    Ok(out)
}

/// Tap a single-T-axis `uniform_tdim` wire (e.g. `chunk_row [T, 1]` or
/// `chunk_col [1, T]`), split its T-axis at `k`, move the chunk axis to
/// position 0, and — for the contracted side — `WindowOnAxis` with a
/// sentinel pad so out-of-stream boundary slots produce out-of-band
/// values for the downstream predicate.  Returns the chunked outlet.
#[allow(clippy::too_many_arguments)]
fn chunkify_uniform_tdim_input(
    patch: &mut TypedModelPatch,
    model: &TypedModel,
    input: OutletId,
    name_prefix: &str,
    mask: &MaskForm,
    contracted_axis: usize,
    chunk_sym: &Symbol,
    k: i64,
) -> TractResult<OutletId> {
    let in_fact = model.outlet_fact(input)?;
    let positions = streaming_positions(in_fact, chunk_sym);
    ensure!(
        positions.len() == 1,
        "uniform_tdim initiator input must have exactly one streaming axis (got {})",
        positions.len(),
    );
    let stream_axis = positions[0];

    let tapped = patch.tap_model(model, input)?;

    // Cast TDim → I64 up-front: PulsePad (used by WindowOnAxis pulsifier
    // for the contracted side) fills with `dispatch_copy_by_size!`, which
    // panics on non-Copy datum types like TDim.  Body ops downstream are
    // Sub/Ge/Le/And — they don't care whether their numeric inputs are
    // TDim or I64 (the final mask comes out as Bool either way).
    //
    // REVISIT: add a TDim arm to `pulse-opl/src/pad.rs` (clone-fill instead
    // of `dispatch_copy_by_size`) and drop this round-trip.
    let mut wire = tapped;
    if patch.outlet_fact(wire)?.datum_type == TDim::datum_type() {
        wire = patch.wire_node(
            format!("{name_prefix}.cast_i64"),
            tract_core::ops::cast::cast(i64::datum_type()),
            &[wire],
        )?[0];
    }
    let dt = patch.outlet_fact(wire)?.datum_type;

    // Split the T-axis at `k`.  Output rank = input rank + 1, with the
    // chunk axis at `stream_axis` and the within-block axis at
    // `stream_axis + 1`.
    wire = wire_chunk_split(patch, name_prefix, wire, stream_axis, chunk_sym, k)?;

    // Move chunk axis to position 0 if it isn't already, so the section
    // frame uniformly carries the chunk axis at 0.
    if stream_axis != 0 {
        wire = patch.wire_node(
            format!("{name_prefix}.move_chunk"),
            AxisOp::Move(stream_axis, 0),
            &[wire],
        )?[0];
    }

    // If this input's source T-axis is the contracted side, window the
    // chunk axis (now at position 0) and flatten the W slot back into the
    // within-block axis so downstream consumers see W·k along that axis.
    let needs_window = !mask.is_block_diag() && stream_axis == contracted_axis;
    if needs_window {
        let window_size: usize = (mask.upper - mask.lower + 1) as usize;
        let start = window_start_for(mask, contracted_axis);
        // Sentinel pad: a value far outside any sane chunk-index range,
        // so the downstream band predicate `chunk_a − chunk_b ∈ [lower,
        // upper]` is false on out-of-stream boundary slots.
        let sentinel = sentinel_pad_value(dt)?.into_arc_tensor();
        wire = patch.wire_node(
            format!("{name_prefix}.window"),
            tract_pulse_opl::ops::WindowOnAxis {
                axis: 0,
                window: window_size,
                start,
                pad_value: sentinel,
            },
            &[wire],
        )?[0];

        // Post-window shape: chunk at 0, W at 1, then the original axes
        // (the within-block axis is at `stream_axis + 1` post-window:
        // chunk was at 0 pre-window, W gets inserted at 1, so axes shift
        // right by 1).  Flatten W (slice index 0) and within-block (slice
        // index `stream_axis`) into a single (W·k) axis.
        let post_window = patch.outlet_fact(wire)?.clone();
        let rank_after = post_window.rank();
        let from: TVec<TDim> = (1..rank_after).map(|i| post_window.shape[i].clone()).collect();
        let within_slice_idx = stream_axis;
        let mut to: TVec<TDim> = tvec!();
        for (i, dim) in from.iter().enumerate() {
            if i == 0 {
                continue;
            }
            if i == within_slice_idx + 1 {
                let merged = from[0].clone() * dim.clone();
                to.push(merged);
            } else {
                to.push(dim.clone());
            }
        }
        wire = patch.wire_node(
            format!("{name_prefix}.flatten_window"),
            AxisOp::Reshape(1, from, to),
            &[wire],
        )?[0];
    }

    Ok(wire)
}

/// Pad value for windowing a chunk-index wire: any value outside any
/// reasonable `[lower, upper]` band so the downstream `Ge`/`Le`
/// comparisons on `chunk_a − chunk_b` evaluate to false at boundary
/// slots.  Bounded by `i32::MAX / 4` because tract's tensor cast routes
/// `i64 → TDim` through `i32` (see `data/src/tensor.rs:1250`), which
/// would truncate a larger sentinel.  Half a billion is comfortably
/// above any plausible chunk count yet safe under that cast.
///
/// REVISIT: route `i64 → TDim` directly in `data/src/tensor.rs:1250`
/// (no `i32` middle step) and lift the cap to `i64::MAX / 4`.
fn sentinel_pad_value(dt: DatumType) -> TractResult<Tensor> {
    if dt == bool::datum_type() {
        bail!("uniform_tdim wire of bool dtype not expected as initiator-side input");
    }
    Ok(tensor0((i32::MAX / 4) as i64).cast_to_dt(dt)?.into_owned())
}

/// Replay a body op in the chunked frame.
///
/// * Inputs from `chunked` (= chunked wires produced by the initiator path
///   for both the data side and the mask-construction chain): pass through.
/// * Other external inputs: tapped, with `AddAxis(0)` bumping any rank
///   deficit so rank-strict consumers (TypedBinOp, …) accept them.
///
/// `axes_mapping::track_axis` asserts each chunked input's chunk axis
/// reaches a unique output axis — bails with a precise error if the op
/// would disconnect the chunk axis (e.g. softmax over it).
#[allow(clippy::too_many_arguments)]
fn wire_body(
    patch: &mut TypedModelPatch,
    model: &TypedModel,
    node: &TypedNode,
    _mask: &MaskForm,
    _contracted_axis: usize,
    chunked: &HashMap<OutletId, OutletId>,
    chunk_sym: &Symbol,
    _k: i64,
) -> TractResult<OutletId> {
    // Pass 1: collect chunked inputs and discover the rank we'll work in.
    // A chunked input may have 1 streaming axis (the usual case — this
    // input contributes a chunk axis we need to track through `op`'s
    // axes_mapping), or 0 streaming axes (rank-bumped broadcast constant
    // produced by `wire_initiator_multibroadcastto` — its value is
    // independent of the chunk index, so no axis-mapping check needed).
    let n = node.inputs.len();
    let mut new_inputs: TVec<Option<OutletId>> = tvec![None; n];
    let mut chunk_input_axes: Vec<(usize, usize)> = vec![];
    let mut chunked_rank: Option<usize> = None;
    for (slot, &input) in node.inputs.iter().enumerate() {
        if let Some(&c) = chunked.get(&input) {
            let cf = patch.outlet_fact(c)?;
            let positions = streaming_positions(cf, chunk_sym);
            ensure!(
                positions.len() <= 1,
                "Body op {node}: chunked input slot {slot} has {} streaming axes, expected ≤ 1",
                positions.len()
            );
            if let Some(&ax) = positions.first() {
                chunk_input_axes.push((slot, ax));
            }
            chunked_rank = Some(cf.rank().max(chunked_rank.unwrap_or(0)));
            new_inputs[slot] = Some(c);
        }
    }
    let chunked_rank = chunked_rank.ok_or_else(|| {
        format_err!("Body op {node} has no chunked input — at least one is required")
    })?;

    // Pass 2: external taps, rank-bumped to match the chunked frame.
    // (uniform_tdim mask wires are now in `chunked` already, courtesy of
    // the uniform_tdim initiator + faithful body chunking — no per-op
    // mask-substitute logic.)
    for (slot, &input) in node.inputs.iter().enumerate() {
        if new_inputs[slot].is_some() {
            continue;
        }
        let tapped = patch.tap_model(model, input)?;
        let bumped = bump_rank_to(patch, &node.name, slot, tapped, chunked_rank)?;
        new_inputs[slot] = Some(bumped);
    }
    let new_inputs: TVec<OutletId> = new_inputs.into_iter().map(|o| o.unwrap()).collect();

    // Chunkability: for every chunked input, its chunk axis must track
    // through to a unique output axis position.
    let input_facts: TVec<TypedFact> =
        new_inputs.iter().map(|o| patch.outlet_fact(*o).cloned()).collect::<TractResult<_>>()?;
    let in_refs: TVec<&TypedFact> = input_facts.iter().collect();
    let output_facts = node.op.output_facts(&in_refs)?;
    let out_refs: TVec<&TypedFact> = output_facts.iter().collect();
    let am = node.op.axes_mapping(&in_refs, &out_refs)?;
    for &(slot, axis) in &chunk_input_axes {
        let tracked = am.track_axis((InOut::In(slot), axis), InOut::Out(0))?;
        ensure!(
            tracked.is_some(),
            "Body op {node} doesn't preserve the chunk axis (input slot {slot}, axis {axis}) \
             through to the output — its axes_mapping disconnects it"
        );
    }

    // Some body ops carry an explicit axis or axes parameter (Softmax,
    // AxisOp::Move/Add/Rm, …) whose values are positions in the *original*
    // rank.  The chunk axis is inserted at the chunked input's chunk
    // position; every original axis at or beyond that position shifts
    // right by one.  Translate accordingly.  When inputs disagree on the
    // chunk position we punt (no consistent chunk_pos to translate
    // against); that case shouldn't arise in a valid section.
    let chunk_pos = chunk_input_axes.iter().map(|&(_, ax)| ax).next();
    if let Some(cp) = chunk_pos {
        ensure!(
            chunk_input_axes.iter().all(|&(_, ax)| ax == cp),
            "Body op {node}: chunked inputs disagree on chunk axis position {chunk_input_axes:?}"
        );
    }
    let chunked_op = translate_body_op_axes(node.op.as_ref(), chunk_pos);
    Ok(patch.wire_node(&*node.name, chunked_op, &new_inputs)?[0])
}

/// Rewrite an op's axis/axes parameters for the chunked frame, where
/// the chunk axis was inserted at `chunk_pos` (taken from the chunked
/// input's streaming axis position).  Original axes at or beyond
/// `chunk_pos` shift right by one; axes strictly before it stay put.
/// Handles `Softmax`, `AxisOp::Move/Add/Rm`; other axis-bearing ops
/// fall through unchanged.
fn translate_body_op_axes(op: &dyn TypedOp, chunk_pos: Option<usize>) -> Box<dyn TypedOp> {
    use tract_core::ops::nn::{Softmax, SoftmaxKind};
    let shift = |a: usize| match chunk_pos {
        Some(cp) => chunked_axis_index(a, cp),
        None => a,
    };
    if let Some(softmax) = op.downcast_ref::<Softmax>() {
        let new_axes: TVec<usize> = softmax.axes.iter().map(|&a| shift(a)).collect();
        let new_softmax = match &softmax.kind {
            SoftmaxKind::Softmax(exp) => {
                Softmax::new(new_axes, softmax.quant_output_dt, SoftmaxKind::Softmax(*exp))
            }
            SoftmaxKind::LogSoftmax => {
                Softmax::new(new_axes, softmax.quant_output_dt, SoftmaxKind::LogSoftmax)
            }
        };
        return Box::new(new_softmax);
    }
    if let Some(ax_op) = op.downcast_ref::<AxisOp>() {
        // `Add(at)` inserts a new axis *before* the original position `at`.
        // We want the new axis to land in the same broadcast slot relative
        // to the original tensor, which means it stays at `at` when
        // `at <= chunk_pos` (placed before the chunk axis) and shifts +1
        // otherwise.  Move/Rm name existing axes, so their parameters
        // translate via `chunked_axis_index` like any other label.
        let add_shift = |a: usize| match chunk_pos {
            Some(cp) if a > cp => a + 1,
            _ => a,
        };
        let translated = match ax_op {
            AxisOp::Move(from, to) => AxisOp::Move(shift(*from), shift(*to)),
            AxisOp::Add(at) => AxisOp::Add(add_shift(*at)),
            AxisOp::Rm(at) => AxisOp::Rm(shift(*at)),
            other => other.clone(),
        };
        return Box::new(translated);
    }
    tract_core::dyn_clone::clone_box(op)
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

#[allow(clippy::too_many_arguments)]
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
#[allow(clippy::too_many_arguments)]
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
    let score_rank = node.outputs[0].fact.rank();
    let rank_diff = score_rank.checked_sub(2).ok_or_else(|| {
        format_err!("Score rank {score_rank} < 2; cannot translate to mask frame")
    })?;
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
        let chunked = wire_chunk_split(
            patch,
            &format!("{}.{ix}", node.name),
            tapped,
            stream_axis,
            chunk_sym,
            k,
        )?;

        // Banded path: if this input's stream axis is on the contracted
        // side of the section, expose `W` chunks per pulse on it.
        // Translate the tracked score axis to mask frame for the
        // comparison with `contracted_axis` (also mask frame).
        let tracked_in_score =
            op.axes.track_axis((InOut::In(ix), stream_axis), InOut::Out(0))?.ok_or_else(|| {
                format_err!(
                    "EinSum stream axis on input {ix} doesn't track to a unique output axis"
                )
            })?;
        let tracked_in_mask = tracked_in_score.checked_sub(rank_diff).ok_or_else(|| {
            format_err!(
                "Tracked score axis {tracked_in_score} doesn't map to mask frame \
                 (rank_diff={rank_diff})"
            )
        })?;
        let chunked = wrap_with_window_if_needed(
            patch,
            chunked,
            stream_axis,
            tracked_in_mask,
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
#[allow(clippy::too_many_arguments)]
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
    let dt = patch.outlet_fact(chunked)?.datum_type;
    let pad_value = Tensor::zero_scalar_dt(dt)?.into_arc_tensor();
    let windowed = patch.wire_node(
        format!("{name_prefix}.window"),
        tract_pulse_opl::ops::WindowOnAxis { axis: stream_axis, window, start, pad_value },
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
#[allow(clippy::too_many_arguments)]
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
    let score_rank = model.outlet_fact(node.inputs[0])?.rank();
    let rank_diff = score_rank.checked_sub(2).ok_or_else(|| {
        format_err!("Terminator score rank {score_rank} < 2; cannot translate to mask frame")
    })?;
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
            let new_chunked = wire_chunk_split(
                patch,
                &format!("{}.in{slot}", node.name),
                tapped,
                stream_axis,
                chunk_sym,
                k,
            )?;

            // Where does this auxiliary's stream axis sit on the score
            // matrix (= input 0 of this einsum)?  If it's the contracted
            // side, window it so its within-chunk axis matches the
            // already-windowed score input.  Translate the score axis
            // to mask frame for the comparison with `contracted_axis`.
            let aux_in_score = op.axes.track_axis((InOut::In(slot), stream_axis), InOut::In(0))?;
            let new_chunked = if let Some(score_axis) = aux_in_score
                && let Some(mask_axis) = score_axis.checked_sub(rank_diff)
            {
                wrap_with_window_if_needed(
                    patch,
                    new_chunked,
                    stream_axis,
                    mask_axis,
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

/// Compute the constant `c` such that `dim == c + k · chunk_sym`, when one
/// exists.  Encoder-style conv stacks emit dims like `1 + (T+6)/8` which,
/// after the `T → P · S` substitute, become `1 + 14·S` — affine in `S`
/// with constant `c = 1`.  Blockify's chunked Reshape can't directly
/// reshape `c + k·S → [S, k]`; we slice off the trailing `c` tokens
/// first so the chunkable region is exactly `k·S`.
///
/// Returns `Some(c)` only when `c` is a non-negative integer constant.
/// `c = 0` is the clean case (no slice needed).
fn affine_chunk_offset(dim: &TDim, chunk_sym: &Symbol, k: i64) -> Option<i64> {
    let target = chunk_sym.to_dim() * k;
    let diff = dim.clone() - target;
    let c = diff.to_i64().ok()?;
    (c >= 0).then_some(c)
}

/// Wrap a chunked `Reshape(stream_axis, [dim], [chunk_sym, k])` with an
/// `AffineChunkTrim` when the input dim is `c + k · chunk_sym` for
/// `c > 0`, dropping the trailing `c` tokens so the Reshape sees `k·S`.
fn wire_chunk_split(
    patch: &mut TypedModelPatch,
    name: &str,
    input: OutletId,
    stream_axis: usize,
    chunk_sym: &Symbol,
    k: i64,
) -> TractResult<OutletId> {
    let in_fact = patch.outlet_fact(input)?.clone();
    let dim = in_fact.shape[stream_axis].clone();
    let target = chunk_sym.to_dim() * k;
    let mut wire = input;
    if dim != target
        && let Some(c) = affine_chunk_offset(&dim, chunk_sym, k)
        && c > 0
    {
        wire = patch.wire_node(
            format!("{name}.affine_trim"),
            crate::ops::array::AffineChunkTrim {
                axis: stream_axis,
                typed_trim: c as usize,
                target_per_pulse: k as usize,
            },
            &[wire],
        )?[0];
    }
    let from = tvec!(patch.outlet_fact(wire)?.shape[stream_axis].clone());
    let to = tvec!(chunk_sym.to_dim(), k.to_dim());
    Ok(patch.wire_node(
        format!("{name}.blockify_split"),
        AxisOp::Reshape(stream_axis, from, to),
        &[wire],
    )?[0])
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
    Ok(EinSum { axes: new_mapping, operating_dt: op.operating_dt, q_params: op.q_params })
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
