//! Transformer-XL relative-position "skew trick" folded into a single op.
//!
//! The skew chain `Pad(axis, pre=1) → Reshape([T,2T]→[2T,T]) → Slice(start=1)
//! → Reshape([2T-1,T]→[T,2T-1]) → Slice(end=T)` converts relative-position
//! scores `[…, T, 2T-1]` into absolute-position scores `[…, T, T]`.  This
//! module replaces that 5-op chain with a single [`DiagGather`] whose
//! per-element semantics are trivial: `output[…, i, k] = input[…, i, offset + k − i]`.
//!
//! Folding is a pure typed-model rewrite (strength reduction): the op is
//! cheap to evaluate, pulsifier-friendly, and easier for downstream passes to
//! reason about than the chain it replaces.

use tract_nnef::internal::*;
use tract_nnef::tract_core::ops::array::{Pad, PadMode, Slice};
use tract_nnef::tract_core::ops::change_axes::{AxisOp, InOut};

/// Diagonal gather: `output[…, i, k] = input[…, i, offset + k − i]`
///
/// `offset` is the centre of the relative-position table (typically `T - 1`)
/// and `out_len` is the number of output columns per query row (typically `T`).
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct DiagGather {
    /// Centre of the relative position table: `T − 1`.
    pub offset: TDim,
    /// Number of output columns per query row.
    pub out_len: TDim,
}

impl Op for DiagGather {
    fn name(&self) -> StaticName {
        "DiagGather".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("offset={}, out_len={}", self.offset, self.out_len)])
    }

    op_as_typed_op!();
}

impl EvalOp for DiagGather {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        _node_id: usize,
        session: &TurnState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let rank = input.rank();
        let t = input.shape()[rank - 2];
        let r = input.shape()[rank - 1];
        let offset = self.offset.eval(&session.resolved_symbols).to_i64()? as isize;
        let out_len = self.out_len.eval(&session.resolved_symbols).to_usize()?;

        let mut out_shape: TVec<usize> = input.shape().into();
        out_shape[rank - 1] = out_len;

        unsafe {
            let mut output = Tensor::uninitialized_dt(input.datum_type(), &out_shape)?;
            let elem_size = input.datum_type().size_of();
            let in_ptr = input.as_ptr_unchecked::<u8>();
            let out_ptr = output.as_ptr_mut_unchecked::<u8>();

            let batch_size: usize = out_shape[..rank - 2].iter().product();
            let in_row_stride = r * elem_size;
            let out_row_stride = out_len * elem_size;

            for b in 0..batch_size {
                for i in 0..t {
                    let in_row = in_ptr.add((b * t + i) * in_row_stride);
                    let out_row = out_ptr.add((b * t + i) * out_row_stride);
                    for k in 0..out_len {
                        let idx = offset + k as isize - i as isize;
                        if idx >= 0 && (idx as usize) < r {
                            std::ptr::copy_nonoverlapping(
                                in_row.add(idx as usize * elem_size),
                                out_row.add(k * elem_size),
                                elem_size,
                            );
                        } else {
                            std::ptr::write_bytes(out_row.add(k * elem_size), 0, elem_size);
                        }
                    }
                }
            }
            Ok(tvec!(output.into_tvalue()))
        }
    }
}

impl TypedOp for DiagGather {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut shape: TVec<TDim> = inputs[0].shape.to_tvec();
        let rank = shape.len();
        shape[rank - 1] = self.out_len.clone();
        Ok(tvec!(inputs[0].datum_type.fact(&shape)))
    }

    fn axes_mapping(
        &self,
        inputs: &[&TypedFact],
        _outputs: &[&TypedFact],
    ) -> TractResult<AxesMapping> {
        // All axes map 1:1 between input and output.
        // The last axis is semantically a gather (not element-wise), but
        // for axis tracking purposes it maps input-last to output-last.
        AxesMapping::natural_for_rank(1, 1, inputs[0].rank())
    }

    fn input_roi(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TVec<Option<TDim>>>> {
        // Output indexing: out[..., q, c] = in[..., q, offset + c - q]
        // So input position (q, r) is read for output position (q, r + q - offset).
        // To translate output ROI to input ROI, substitute the c symbol with
        // (r + q - offset) where r is the input's last axis and q is the
        // shared axis at rank-2.
        let output_fact = model.outlet_fact(OutletId::new(node.id, 0))?;
        let Some(roi) = &output_fact.region_of_interest else { return Ok(None) };
        let rank = output_fact.shape.rank();
        if rank < 2 {
            return Ok(Some(tvec![Some(roi.clone())]));
        }
        let c_sym = roi
            .symbols()
            .into_iter()
            .find(|s| tract_nnef::tract_core::ops::logic::sym_to_coord_axis(s) == Some(rank - 1));
        let Some(c_sym) = c_sym else {
            // No mention of the column axis — pass through unchanged.
            return Ok(Some(tvec![Some(roi.clone())]));
        };
        let Some(scope) = c_sym.scope() else { return Ok(Some(tvec![Some(roi.clone())])) };
        let q_sym = TDim::Sym(scope.coord_sym(rank - 2));
        let r_expr = TDim::Sym(c_sym.clone()) + q_sym - self.offset.clone();
        let input_roi = roi.substitute(&c_sym, &r_expr).map(|d| d.reduce()).unwrap_or(roi.clone());
        Ok(Some(tvec![Some(input_roi)]))
    }

    fn set_symbols(
        &self,
        _source: &TypedModel,
        node: &TypedNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
        subs: &HashMap<Symbol, TDim>,
    ) -> TractResult<TVec<OutletId>> {
        let inputs = node.inputs.iter().map(|i| mapping[i]).collect::<TVec<_>>();
        let op = DiagGather {
            offset: self.offset.substitute_all(subs)?,
            out_len: self.out_len.substitute_all(subs)?,
        };
        target.wire_node(&node.name, op, &inputs)
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        declutter_narrow_via_band_roi(self, model, node)
    }

    as_op!();
}

/// Coordinated narrow: when an axes-preserving chain upstream of this
/// DiagGather terminates at a `Slice` whose output has a constant-width
/// band ROI on the slice axis, narrow the Slice to that band AND re-anchor
/// `DG.offset` so the rel-pos-zero column stays at the right place.
///
/// Math: with `lo, hi_excl` the band bounds on the slice's output axis,
/// `new_slice.start = old_slice.start + lo`, `new_slice.end =
/// old_slice.start + hi_excl`, `new_offset = old_offset − lo`.  All three
/// must simplify to concrete integers before applying.
fn declutter_narrow_via_band_roi(
    op: &DiagGather,
    model: &TypedModel,
    node: &TypedNode,
) -> TractResult<Option<TypedModelPatch>> {
    // The rel-pos axis on DG's input is the last axis.
    let dg_input_rank = model.outlet_fact(node.inputs[0])?.shape.rank();
    if dg_input_rank < 1 {
        return Ok(None);
    }
    let rel_pos_axis = dg_input_rank - 1;

    let Some(trace) = trace_back_to_slice(model, node.inputs[0], rel_pos_axis)? else {
        return Ok(None);
    };
    let slice_node = &model.nodes()[trace.slice_id];
    let Some(slice_op) = slice_node.op_as::<Slice>() else { return Ok(None) };
    let slice_fact = model.outlet_fact(OutletId::new(trace.slice_id, 0))?;
    let Some(roi) = &slice_fact.region_of_interest else { return Ok(None) };
    let scope = model.symbols.clone();
    let axis_sym = scope.coord_sym(slice_op.axis);
    let Some((lo, hi_excl)) = bounds_on_axis_tdim(roi, &axis_sym) else {
        return Ok(None);
    };

    // Compute new slice bounds.  Both must reduce to concrete integers
    // (the downstream chain's `output_facts` re-derives the chain shape
    // from the narrowed slice, so symbolic bounds would leave it stuck).
    let new_start = (slice_op.start.clone() + lo.clone()).reduce();
    let new_end = (slice_op.start.clone() + hi_excl.clone()).reduce();
    if new_start.as_i64().is_none() || new_end.as_i64().is_none() {
        return Ok(None);
    }
    if new_start == slice_op.start && new_end == slice_op.end {
        return Ok(None); // No narrowing.
    }

    // Re-anchor DG.offset.  `op.offset` corresponds to "centre − slice.start"
    // in absolute pos_enc rows; narrowing shifts slice.start by lo, so
    // new_offset = old_offset − lo.
    let new_offset_tdim = (op.offset.clone() - lo).reduce();
    let Ok(new_offset) = new_offset_tdim.to_i64() else { return Ok(None) };

    // Build coordinated patch: re-wire slice with narrow bounds, replay
    // intermediate chain nodes, then wire new DG with adjusted offset.
    let mut patch = TypedModelPatch::new(format!("narrow_via_band_roi@{}", node.name));
    let src = patch.tap_model(model, slice_node.inputs[0])?;
    let new_slice = Slice { axis: slice_op.axis, start: new_start, end: new_end };
    let mut current =
        patch.wire_node(format!("{}.narrowed", slice_node.name), new_slice, &[src])?[0];

    // `trace.intermediate` is the chain from slice's successor up to (but
    // not including) DG itself, ordered from upstream to downstream.
    for (chain_nid, path_in_idx) in &trace.intermediate {
        let chain_node = &model.nodes()[*chain_nid];
        let mut new_inputs: TVec<OutletId> = tvec!();
        for (i, inp) in chain_node.inputs.iter().enumerate() {
            if i == *path_in_idx {
                new_inputs.push(current);
            } else {
                new_inputs.push(patch.tap_model(model, *inp)?);
            }
        }
        current = patch.wire_node(
            format!("{}.narrow_replay", chain_node.name),
            chain_node.op.clone(),
            &new_inputs,
        )?[0];
    }

    let new_dg = DiagGather { offset: TDim::Val(new_offset), out_len: op.out_len.clone() };
    let new_dg_out =
        patch.wire_node(format!("{}.narrowed_offset", node.name), new_dg, &[current])?[0];
    patch.shunt_outside(model, OutletId::new(node.id, 0), new_dg_out)?;
    Ok(Some(patch))
}

/// Result of walking backward from a DiagGather along the rel-pos axis
/// through axes-mapping-preserving ops until a `Slice` is reached.
struct ReverseTrace {
    slice_id: usize,
    /// Chain nodes between (exclusive) Slice and (exclusive) DG, ordered
    /// upstream→downstream, with the input index that carries the rel-pos
    /// axis at each step.
    intermediate: Vec<(usize, usize)>,
}

fn trace_back_to_slice(
    model: &TypedModel,
    start_outlet: OutletId,
    start_axis: usize,
) -> TractResult<Option<ReverseTrace>> {
    let mut current_outlet = start_outlet;
    let mut current_axis = start_axis;
    let mut intermediate: Vec<(usize, usize)> = vec![];
    for _ in 0..32 {
        let node = &model.nodes()[current_outlet.node];
        if let Some(slice_op) = node.op_as::<Slice>()
            && slice_op.axis == current_axis
        {
            intermediate.reverse();
            return Ok(Some(ReverseTrace { slice_id: node.id, intermediate }));
        }
        let input_facts: TVec<&TypedFact> =
            node.inputs.iter().map(|inp| model.outlet_fact(*inp)).collect::<TractResult<_>>()?;
        let output_facts: TVec<&TypedFact> = node.outputs.iter().map(|o| &o.fact).collect();
        let Ok(mapping) = node.op.axes_mapping(&input_facts, &output_facts) else {
            return Ok(None);
        };
        let mut advanced: Option<(usize, usize)> = None;
        for (i, _inp) in node.inputs.iter().enumerate() {
            let Some(in_axis) = mapping
                .track_axis((InOut::Out(current_outlet.slot), current_axis), InOut::In(i))?
            else {
                continue;
            };
            // Only follow inputs whose axis size matches output axis size on this axis.
            let in_fact = &input_facts[i];
            if in_fact.shape[in_axis] == node.outputs[current_outlet.slot].fact.shape[current_axis]
            {
                advanced = Some((i, in_axis));
                break;
            }
        }
        let Some((idx, ax)) = advanced else { return Ok(None) };
        intermediate.push((current_outlet.node, idx));
        current_outlet = node.inputs[idx];
        current_axis = ax;
    }
    Ok(None)
}

/// Extract `(lo, hi_excl)` TDim bounds from a band ROI predicate of shape
/// `Mul(Ge(hi, 🎯_axis), Ge(🎯_axis, lo))` (either order).  The returned
/// `hi_excl = hi + 1` matches the half-open `[lo, hi_excl)` convention
/// used by `Slice::{start, end}`.
fn bounds_on_axis_tdim(roi: &TDim, axis_sym: &Symbol) -> Option<(TDim, TDim)> {
    let TDim::Mul(terms) = roi else { return None };
    if terms.len() != 2 {
        return None;
    }
    let mut lo: Option<TDim> = None;
    let mut hi: Option<TDim> = None;
    for term in terms {
        let TDim::Ge(left, right) = term else { return None };
        if let TDim::Sym(s) = left.as_ref()
            && s == axis_sym
        {
            lo = Some((**right).clone());
            continue;
        }
        if let TDim::Sym(s) = right.as_ref()
            && s == axis_sym
        {
            hi = Some((**left).clone());
            continue;
        }
        return None;
    }
    Some((lo?, hi? + TDim::Val(1)))
}

// ─── Detect pass: Pad → Reshape → Slice → Reshape → Slice  →  DiagGather ──

/// Scan the model for skew-trick chains and replace each with a single
/// [`DiagGather`].
pub fn detect_diag_gather(model: &mut TypedModel) -> TractResult<()> {
    Rewriter::default()
        .with_rule_for::<Pad>("detect-diag-gather", diag_gather_rule)
        .rewrite(&(), model)
}

/// Rewrite rule fired by `Rewriter` on each `Pad` node — matches the
/// `Pad → Reshape → Slice → Reshape → Slice` skew chain anchored at this
/// `Pad` and replaces it with a single `DiagGather`.
pub fn diag_gather_rule(
    _ctx: &(),
    model: &TypedModel,
    pad_node: &TypedNode,
    _node_name: &str,
    pad_op: &Pad,
) -> TractResult<Option<TypedModelPatch>> {
    // ── Step 1: Pad must be Constant(0), one axis (pre=1, post=0), last axis ─
    rule_if_let!(PadMode::Constant(c) = &pad_op.mode);
    rule_if!(c.cast_to_scalar::<f64>().ok() == Some(0.0));
    rule_if_some!(pad_axis = pad_op.pads.iter().position(|&(a, b)| a != 0 || b != 0));
    rule_if!(pad_op.pads[pad_axis] == (1, 0));
    rule_if!(
        !pad_op.pads.iter().enumerate().any(|(i, &(a, b))| i != pad_axis && (a != 0 || b != 0))
    );
    let rank = model.outlet_fact(pad_node.inputs[0])?.rank();
    rule_if!(pad_axis == rank - 1);

    // ── Step 2: Pad → Reshape (transpose last two axes) ────────────────────
    rule_if_some!(reshape1_node = model.single_succ(pad_node.id)?);
    rule_if_let!(Some(AxisOp::Reshape(at1, from1, to1)) = reshape1_node.op_as::<AxisOp>());
    rule_if!(from1.len() == 2 && to1.len() == 2);
    // Block must cover (query axis, padded axis); at1 = rank-2, pad_axis = rank-1.
    rule_if!(*at1 + 1 == pad_axis);
    // Verify transpose: from=[D1, D2] to=[D2, D1].
    rule_if!(from1[0] == to1[1] && from1[1] == to1[0]);
    let d1 = &from1[0]; // query dim (T)

    // ── Step 3: Reshape → Slice (drop the leading padded row) ──────────────
    rule_if_some!(slice1_node = model.single_succ(reshape1_node.id)?);
    rule_if_some!(slice1_op = slice1_node.op_as::<Slice>());
    rule_if!(slice1_op.axis == *at1 && slice1_op.start == 1.to_dim());

    // ── Step 4: Slice → Reshape (transpose back) ───────────────────────────
    rule_if_some!(reshape2_node = model.single_succ(slice1_node.id)?);
    rule_if_let!(Some(AxisOp::Reshape(at2, from2, to2)) = reshape2_node.op_as::<AxisOp>());
    rule_if!(from2.len() == 2 && to2.len() == 2);
    rule_if!(*at2 == *at1);
    // Inverse transpose: from=[D2-1, D1] to=[D1, D2-1].
    rule_if!(from2[0] == to2[1] && from2[1] == to2[0]);
    rule_if!(from2[1] == *d1);

    // ── Step 5: Reshape → Slice (take first D1 columns) ────────────────────
    rule_if_some!(slice2_node = model.single_succ(reshape2_node.id)?);
    rule_if_some!(slice2_op = slice2_node.op_as::<Slice>());
    rule_if!(slice2_op.axis == at2 + 1 && slice2_op.start == 0.to_dim());

    // ── Build the replacement DiagGather ────────────────────────────────────
    let diag_gather = DiagGather {
        offset: d1.clone() - 1,                            // T - 1
        out_len: slice2_op.end.clone() - &slice2_op.start, // = D1
    };

    let mut patch = TypedModelPatch::new("detect-diag-gather");
    let pos_raw = patch.tap_model(model, pad_node.inputs[0])?;
    let out = patch.wire_node(&slice2_node.name, diag_gather, &[pos_raw])?[0];
    patch.shunt_outside(model, slice2_node.id.into(), out)?;

    Ok(Some(patch))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build the skew trick chain and verify DiagGather fold produces correct output.
    #[test]
    fn test_detect_diag_gather_concrete() -> TractResult<()> {
        let t: usize = 4;
        let r = 2 * t - 1; // 7

        // Build a model with the skew trick chain.
        let mut model = TypedModel::default();
        let input = model.add_source("pos_raw", f32::fact(&[1, t, r]))?;

        // Pad axis 2, pre=1
        let mut pads = vec![(0, 0); 3];
        pads[2] = (1, 0);
        let padded = model.wire_node(
            "pad",
            Pad::new(pads, PadMode::Constant(rctensor0(0.0f32))),
            &[input],
        )?[0];

        // Reshape [T, 2T] → [2T, T]
        let reshaped1 = model.wire_node(
            "reshape1",
            AxisOp::Reshape(
                1,
                tvec![t.to_dim(), (2 * t).to_dim()],
                tvec![(2 * t).to_dim(), t.to_dim()],
            ),
            &[padded],
        )?[0];

        // Slice axis=1, start=1, end=2T
        let sliced1 = model.wire_node("slice1", Slice::new(1, 1, 2 * t), &[reshaped1])?[0];

        // Reshape [2T-1, T] → [T, 2T-1]
        let reshaped2 = model.wire_node(
            "reshape2",
            AxisOp::Reshape(
                1,
                tvec![(2 * t - 1).to_dim(), t.to_dim()],
                tvec![t.to_dim(), (2 * t - 1).to_dim()],
            ),
            &[sliced1],
        )?[0];

        // Slice axis=2, start=0, end=T
        let sliced2 = model.wire_node("slice2", Slice::new(2, 0, t), &[reshaped2])?[0];

        model.select_output_outlets(&[sliced2])?;

        // Run the original model.
        let mut rng = 42u64;
        let input_data: Vec<f32> = (0..(t * r))
            .map(|_| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                (rng >> 33) as f32 / 1000.0
            })
            .collect();
        let input_tensor = tensor1(&input_data).into_shape(&[1, t, r])?;
        let original_output =
            model.clone().into_runnable()?.run(tvec![input_tensor.clone().into()])?;

        // Fold.
        let mut folded = model.clone();
        detect_diag_gather(&mut folded)?;

        // Verify the folded model has a DiagGather node.
        assert!(
            folded.nodes().iter().any(|n| n.op_as::<DiagGather>().is_some()),
            "folded model should contain DiagGather"
        );

        // Run the folded model.
        let folded_output = folded.into_runnable()?.run(tvec![input_tensor.into()])?;

        // Compare outputs.
        let orig = original_output[0].to_plain_array_view::<f32>()?;
        let fold = folded_output[0].to_plain_array_view::<f32>()?;
        assert_eq!(orig.shape(), fold.shape());
        for (a, b) in orig.iter().zip(fold.iter()) {
            assert!((*a - *b).abs() < 1e-6, "Mismatch: original={a}, folded={b}");
        }
        Ok(())
    }

    /// DiagGather's `input_roi` should substitute the column axis `c` with
    /// `r + q - offset`: the output index `(q, c)` reads input index
    /// `(q, offset + c - q)`, so input position `(q, r)` matters iff there's
    /// some output `(q, c)` with `r = offset + c - q`, i.e. `c = r + q - offset`.
    ///
    /// Test case: a diagonal-of-width-3 band on the output `(q, c)` —
    /// `Mul(Ge(c, q-1), Ge(q+1, c))` — should translate to a CONSTANT band
    /// `2 ≤ r ≤ 4` on the input (q drops out), because the bandwidth is the
    /// same offset around the diagonal.
    #[test]
    fn diag_gather_input_roi_substitutes_diagonal_coord() -> TractResult<()> {
        let t: usize = 4;
        let r = 2 * t - 1; // 7

        let mut model = TypedModel::default();
        let src = model.add_source("src", f32::fact(&[1, t, r]))?;
        let dg = model.wire_node(
            "dg",
            DiagGather { offset: (t as i64 - 1).to_dim(), out_len: t.to_dim() },
            &[src],
        )?[0];
        model.select_output_outlets(&[dg])?;

        // Plant a diagonal band ROI on dg's output: |q - c| <= 1.
        // That is: Ge(c, q - 1) AND Ge(q + 1, c).
        let q_sym = model.symbols.coord_sym(1);
        let c_sym = model.symbols.coord_sym(2);
        let q = TDim::Sym(q_sym);
        let c = TDim::Sym(c_sym);
        let band = TDim::Mul(vec![
            TDim::Ge(Box::new(c.clone()), Box::new(q.clone() - TDim::Val(1))),
            TDim::Ge(Box::new(q + TDim::Val(1)), Box::new(c)),
        ]);
        model.nodes_mut()[dg.node].outputs[0].fact.region_of_interest = Some(band);

        // Call input_roi on the DG node and inspect what gets planted on input 0.
        let dg_node = &model.nodes()[dg.node];
        let input_rois = dg_node.op.as_typed().unwrap().input_roi(&model, dg_node)?;
        let input_rois = input_rois.expect("DG should return Some");
        let input_roi = input_rois[0].as_ref().expect("DG should plant on input 0");

        // Verify the substitution actually happened: `c` (🎯2) should now
        // appear as the sum `🎯1 + 🎯2 - 3` in both Ge terms.
        let printed = format!("{input_roi}");
        eprintln!("DG input ROI: {printed}");
        assert!(
            printed.contains("🎯1+🎯2+-3") || printed.contains("🎯1+🎯2-3"),
            "expected `c → r + q - offset` substitution, got {printed}"
        );
        Ok(())
    }
}
