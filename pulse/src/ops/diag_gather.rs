use crate::internal::*;
use crate::model::PulseWrappingOp;
use tract_core::ops::array::{Pad, PadMode, Slice};
use tract_core::ops::change_axes::AxisOp;
use tract_core::ops::logic::sym_to_coord_axis;

register_all!(DiagGather: pulsify_diag_gather);

/// If `expr` is a 2-D chunk-window `uniform_tdim` mask, return the live
/// window width `W = (L+1)·P` — the per-row output length DiagGather should
/// produce in pulsed form.
///
/// The mask shape is
/// `Ge(Val(L), diff) * Ge(diff, Val(0))` with
/// `diff = floor((🎯row + r_off) / P) - floor((🎯col + c_off) / P) + constant`.
///
/// Semantically: a query at chunk-index `c = floor(i/P)` attends to keys in
/// chunks `c-L..=c`, i.e. `(L+1)·P` consecutive columns.  That width is the
/// only signal the DiagGather pulsifier needs from the ROI — it does not
/// care how the window decomposes into `L` and `P` individually.
fn chunk_window_width(expr: &TDim) -> Option<u64> {
    let TDim::Mul(factors) = expr else { return None };
    let n = factors.len();
    if n < 2 {
        return None;
    }
    for f0 in 0..n {
        for f1 in 0..n {
            if f0 == f1 {
                continue;
            }
            let TDim::Ge(lhs0, rhs0) = &factors[f0] else { continue };
            let TDim::Ge(lhs1, rhs1) = &factors[f1] else { continue };
            let TDim::Val(l) = lhs0.as_ref() else { continue };
            let TDim::Val(0) = rhs1.as_ref() else { continue };
            let Some((row, col, p)) = extract_div_diff_axes(rhs0) else { continue };
            let Some((row2, col2, p2)) = extract_div_diff_axes(lhs1) else { continue };
            if row != row2 || col != col2 || p != p2 {
                continue;
            }
            if *l < 0 {
                continue;
            }
            return Some((*l as u64 + 1) * p);
        }
    }
    None
}

/// Try to decompose `expr` as a chunk-index difference:
///   `floor((🎯row + r_off) / P) - floor((🎯col + c_off) / P) + constant`
fn extract_div_diff_axes(expr: &TDim) -> Option<(usize, usize, u64)> {
    let TDim::Add(terms) = expr else { return None };
    let mut pos: Option<(usize, u64)> = None;
    let mut neg: Option<(usize, u64)> = None;
    for term in terms {
        match term {
            TDim::Div(inner, p) => {
                if let Some(axis) = extract_coord_sym_from_div_arg(inner) {
                    pos = Some((axis, *p));
                }
            }
            TDim::MulInt(-1, inner) => {
                if let TDim::Div(inner2, p) = inner.as_ref() {
                    if let Some(axis) = extract_coord_sym_from_div_arg(inner2) {
                        neg = Some((axis, *p));
                    }
                }
            }
            TDim::Val(_) => {}
            _ => return None,
        }
    }
    let (row_axis, p_row) = pos?;
    let (col_axis, p_col) = neg?;
    if p_row != p_col {
        return None;
    }
    Some((row_axis, col_axis, p_row))
}

fn extract_coord_sym_from_div_arg(inner: &TDim) -> Option<usize> {
    match inner {
        TDim::Sym(sym) => sym_to_coord_axis(sym),
        TDim::Add(terms) => {
            let mut axis = None;
            for t in terms {
                match t {
                    TDim::Sym(sym) => {
                        if axis.is_some() {
                            return None;
                        }
                        axis = sym_to_coord_axis(sym);
                    }
                    TDim::Val(_) => {}
                    _ => return None,
                }
            }
            axis
        }
        _ => None,
    }
}

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

    fn substitute_symbols(
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
    // Pulse-time `out_len` is the live-window width W from the output's
    // chunk-window ROI.  If the ROI is missing or doesn't match the
    // chunk-window pattern, defer to the regular fallback.
    let roi_raw = source.outlet_fact(OutletId::new(node.id, 0))?.region_of_interest.clone();
    rule_if_some!(w = roi_raw.as_ref().and_then(|r| chunk_window_width(&r.clone().simplify())));

    let input_wire = mapping[&node.inputs[0]];
    let input_fact = target.outlet_fact(input_wire)?.clone();
    let stream = input_fact.stream.as_ref().context("DiagGather input must be streaming")?;

    // P_local: the pulse size at this level (after any subsampling).  In the
    // windowed input the relative-position axis has `W + P_local − 1`
    // entries; distance 0 sits at position `P_local − 1`.
    let p_local = input_fact.shape[stream.axis].to_i64()?;

    let pulsed_op = DiagGather { offset: (p_local - 1).to_dim(), out_len: (w as i64).to_dim() };

    let out = target.wire_node(&node.name, PulseWrappingOp(Box::new(pulsed_op)), &[input_wire])?;
    Ok(Some(out))
}

// ─── Fold pass: Pad → Reshape → Slice → Reshape → Slice  →  DiagGather ─────

/// Scan the model for skew-trick chains and replace them with DiagGather.
///
/// Called by the pulse transform before pulsification.
pub fn detect_diag_gather(model: &mut TypedModel) -> TractResult<()> {
    Rewriter::default()
        .with_rule_for::<Pad>("detect-diag-gather", diag_gather_rule)
        .rewrite(&(), model)
}

/// Rewrite rule fired by `Rewriter` on each `Pad` node — matches the
/// `Pad → Reshape → Slice → Reshape → Slice` skew chain anchored at this
/// `Pad` and replaces it with a single `DiagGather`.
fn diag_gather_rule(
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
}
