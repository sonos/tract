use crate::internal::*;
use crate::model::PulseWrappingOp;
use tract_core::ops::array::{Pad, PadMode, Slice};
use tract_core::ops::change_axes::AxisOp;
use tract_core::ops::logic::classify_chunk_window;

register_all!(DiagGather: pulsify_diag_gather);

/// Diagonal gather: `output[…, i, k] = input[…, i, offset + k − i]`
///
/// This is the algebraic composition of the "skew trick" used to convert
/// relative position scores `[…, T, 2T−1]` into absolute position scores
/// `[…, T, T]`.  The typical skew chain is:
///
///     Pad(axis, pre=1) → Reshape([T,2T]→[2T,T]) → Slice(start=1)
///                       → Reshape([2T−1,T]→[T,2T−1]) → Slice(end=T)
///
/// DiagGather replaces the entire chain with a single op whose per-element
/// semantics are trivial to pulsify.
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
        let output_fact = model.outlet_fact(OutletId::new(node.id, 0))?;
        let Some(roi) = &output_fact.region_of_interest else { return Ok(None) };
        // Pass the output ROI to the input (same coordinate structure for query axis).
        Ok(Some(tvec![Some(roi.clone())]))
    }

    as_op!();
}

// ─── Pulsifier ──────────────────────────────────────────────────────────────

fn pulsify_diag_gather(
    _op: &DiagGather,
    source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
    _symbol: &Symbol,
    _pulse: &TDim,
) -> TractResult<Option<TVec<OutletId>>> {
    // Require a chunk-window ROI on the output.
    let roi_raw = source.outlet_fact(OutletId::new(node.id, 0))?.region_of_interest.clone();
    let roi = match roi_raw.as_ref().and_then(|r| classify_chunk_window(&r.clone().simplify())) {
        Some(p) => p,
        None => return Ok(None),
    };

    let input_wire = mapping[&node.inputs[0]];
    let input_fact = target.outlet_fact(input_wire)?.clone();
    let stream = input_fact.stream.as_ref().context("DiagGather input must be streaming")?;

    // P_local: the pulse size at this level (after any subsampling).
    let p_local = input_fact.shape[stream.axis].to_i64()?;

    let w = (roi.left_chunks as i64 + 1) * roi.p as i64; // window width

    // In the windowed input, the relative-position axis has W + P_local − 1 entries.
    // Distance 0 is at position P_local − 1.
    let pulsed_op = DiagGather { offset: (p_local - 1).to_dim(), out_len: w.to_dim() };

    let out = target.wire_node(&node.name, PulseWrappingOp(Box::new(pulsed_op)), &[input_wire])?;
    Ok(Some(out))
}

// ─── Fold pass: Pad → Reshape → Slice → Reshape → Slice  →  DiagGather ─────

/// Scan the model for skew-trick chains and replace them with DiagGather.
///
/// Called by the pulse transform before pulsification.
pub fn fold_diag_gather(model: &mut TypedModel) -> TractResult<bool> {
    let mut changed = false;
    loop {
        let order = model.eval_order()?;
        let mut patch = None;
        for &nid in &order {
            if let Some(p) = try_fold_at(model, nid)? {
                patch = Some(p);
                break;
            }
        }
        if let Some(p) = patch {
            p.apply(model)?;
            changed = true;
        } else {
            break;
        }
    }
    Ok(changed)
}

/// Try to match a skew-trick chain starting at `pad_id` (a Pad node).
fn try_fold_at(model: &TypedModel, pad_id: usize) -> TractResult<Option<TypedModelPatch>> {
    let pad_node = model.node(pad_id);

    // ── Step 1: Match Pad ──────────────────────────────────────────────────
    let Some(pad_op) = pad_node.op_as::<Pad>() else { return Ok(None) };
    // Must be Constant(0) padding.
    let PadMode::Constant(ref c) = pad_op.mode else { return Ok(None) };
    if c.cast_to_scalar::<f64>().ok() != Some(0.0) {
        return Ok(None);
    }
    // Exactly one axis padded, with (pre=1, post=0).
    let pad_axis = pad_op.pads.iter().position(|&(a, b)| a != 0 || b != 0);
    let Some(pad_axis) = pad_axis else { return Ok(None) };
    if pad_op.pads[pad_axis] != (1, 0) {
        return Ok(None);
    }
    // No other axis padded.
    if pad_op.pads.iter().enumerate().any(|(i, &(a, b))| i != pad_axis && (a != 0 || b != 0)) {
        return Ok(None);
    }

    let pad_input_fact = model.outlet_fact(pad_node.inputs[0])?;
    let rank = pad_input_fact.rank();

    // pad_axis must be the last axis (the relative-position axis).
    if pad_axis != rank - 1 {
        return Ok(None);
    }

    // ── Step 2: Pad → Reshape ──────────────────────────────────────────────
    let Some(reshape1_node) = model.single_succ(pad_id)? else { return Ok(None) };
    let Some(AxisOp::Reshape(at1, from1, to1)) = reshape1_node.op_as::<AxisOp>() else {
        return Ok(None);
    };
    // Must be a 2→2 reshape that "transposes" the last two axes.
    if from1.len() != 2 || to1.len() != 2 {
        return Ok(None);
    }
    // The reshape block must cover the query axis and the padded axis.
    if *at1 + 1 != pad_axis {
        // at1 should be rank-2, pad_axis should be rank-1
        return Ok(None);
    }
    // Verify it's a transpose: from=[D1, D2] to=[D2, D1].
    if from1[0] != to1[1] || from1[1] != to1[0] {
        return Ok(None);
    }
    let d1 = &from1[0]; // query dim (T)
    let _d2 = &from1[1]; // padded rel-pos dim (2T)

    // ── Step 3: Reshape → Slice (remove first row) ─────────────────────────
    let Some(slice1_node) = model.single_succ(reshape1_node.id)? else { return Ok(None) };
    let Some(slice1_op) = slice1_node.op_as::<Slice>() else { return Ok(None) };
    // Must slice on the same axis that the reshape put the padded dim on (= at1).
    if slice1_op.axis != *at1 {
        return Ok(None);
    }
    // Start must be 1 (remove the first row introduced by the pad).
    if slice1_op.start != 1.to_dim() {
        return Ok(None);
    }

    // ── Step 4: Slice → Reshape (transpose back) ───────────────────────────
    let Some(reshape2_node) = model.single_succ(slice1_node.id)? else { return Ok(None) };
    let Some(AxisOp::Reshape(at2, from2, to2)) = reshape2_node.op_as::<AxisOp>() else {
        return Ok(None);
    };
    if from2.len() != 2 || to2.len() != 2 {
        return Ok(None);
    }
    if *at2 != *at1 {
        return Ok(None);
    }
    // Must be the inverse transpose: from=[D2-1, D1] to=[D1, D2-1].
    if from2[0] != to2[1] || from2[1] != to2[0] {
        return Ok(None);
    }
    // Verify consistency: from2[1] (= to2[0]) should be D1.
    if from2[1] != *d1 {
        return Ok(None);
    }

    // ── Step 5: Reshape → Slice (take first D1 columns) ────────────────────
    let Some(slice2_node) = model.single_succ(reshape2_node.id)? else { return Ok(None) };
    let Some(slice2_op) = slice2_node.op_as::<Slice>() else { return Ok(None) };
    // Must slice on the last axis (at2 + 1).
    if slice2_op.axis != at2 + 1 {
        return Ok(None);
    }
    // Start must be 0.
    if slice2_op.start != 0.to_dim() {
        return Ok(None);
    }

    // ── Build the replacement DiagGather ────────────────────────────────────
    let offset = d1.clone() - 1; // T - 1
    let out_len = slice2_op.end.clone() - &slice2_op.start; // should be D1

    let diag_gather = DiagGather { offset, out_len };

    // Wire: take the Pad's input (pos_raw) and pipe through DiagGather.
    let mut patch = TypedModelPatch::new("fold-diag-gather");
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
    fn test_fold_diag_gather_concrete() -> TractResult<()> {
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
            .map(|i| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                (rng >> 33) as f32 / 1000.0
            })
            .collect();
        let input_tensor = tensor1(&input_data).into_shape(&[1, t, r])?;
        let original_output =
            model.clone().into_runnable()?.run(tvec![input_tensor.clone().into()])?;

        // Fold.
        let mut folded = model.clone();
        let did_fold = fold_diag_gather(&mut folded)?;
        assert!(did_fold, "fold_diag_gather should have matched");

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
}
