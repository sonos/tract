use crate::internal::*;
use crate::num_traits::Zero;

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct Slice {
    pub axis: usize,
    pub start: TDim,
    pub end: TDim,
}

impl Slice {
    pub fn new(axis: usize, start: impl ToDim, end: impl ToDim) -> Slice {
        Slice { axis, start: start.to_dim(), end: end.to_dim() }
    }

    pub fn suffix(&self, name: &str) -> String {
        format!("{}.axis{}_{}_{}", name, self.axis, self.start, self.end)
    }

    /// Rewrite `self` as `Concat([zeros, narrow_slice, zeros], axis=self.axis)`
    /// when the output carries a single-axis band ROI on `self.axis`.  Output
    /// shape stays the same (= `self.end − self.start`), so `shunt_outside`
    /// accepts.  The narrow part actually does work; the zero pads are inert
    /// fillers that the downstream consumer's own narrow-via-Concat rewrite
    /// eats via `TypedConcat::slice`.
    pub fn declutter_narrow_via_roi_concat(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let output_fact = model.outlet_fact(OutletId::new(node.id, 0))?;
        let Some(roi) = &output_fact.region_of_interest else { return Ok(None) };
        let axis_sym = model.symbols.coord_sym(self.axis);
        let Some((lo, hi)) = bounds_on_axis(roi, &axis_sym) else { return Ok(None) };
        let out_width = match (self.end.clone() - &self.start).to_i64() {
            Ok(w) => w,
            Err(_) => return Ok(None),
        };
        let lo = lo.max(0);
        let hi = hi.min(out_width);
        if lo == 0 && hi == out_width {
            return Ok(None); // Nothing to narrow.
        }
        if lo >= hi {
            return Ok(None); // Degenerate.
        }

        let mut full_shape: TVec<usize> =
            output_fact.shape.iter().map(|d| d.to_usize()).collect::<TractResult<_>>()?;
        let dt = output_fact.datum_type;

        let mut patch = TypedModelPatch::new(format!("narrow_via_roi@{}", node.name));
        let src = patch.tap_model(model, node.inputs[0])?;
        let narrow_slice = Slice {
            axis: self.axis,
            start: self.start.clone() + TDim::Val(lo),
            end: self.start.clone() + TDim::Val(hi),
        };
        let narrow_out = patch.wire_node(format!("{}.narrow", node.name), narrow_slice, &[src])?[0];

        let mut concat_inputs: TVec<OutletId> = tvec!();
        if lo > 0 {
            full_shape[self.axis] = lo as usize;
            let z = Tensor::zero_dt(dt, &full_shape)?.into_arc_tensor();
            concat_inputs.push(patch.add_const(format!("{}.narrow_zeros_left", node.name), z)?);
        }
        concat_inputs.push(narrow_out);
        if hi < out_width {
            full_shape[self.axis] = (out_width - hi) as usize;
            let z = Tensor::zero_dt(dt, &full_shape)?.into_arc_tensor();
            concat_inputs.push(patch.add_const(format!("{}.narrow_zeros_right", node.name), z)?);
        }
        let concat_out = patch.wire_node(
            format!("{}.narrow_concat", node.name),
            crate::ops::array::TypedConcat::new(self.axis),
            &concat_inputs,
        )?[0];
        patch.shunt_outside(model, OutletId::new(node.id, 0), concat_out)?;
        Ok(Some(patch))
    }

    pub fn declutter_slice_after_slice(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let prec = model.node(node.inputs[0].node);
        if let Some(other) = prec.op_as::<Slice>()
            && other.axis == self.axis
        {
            return TypedModelPatch::replace_single_op(
                model,
                node,
                &prec.inputs,
                Slice {
                    axis: self.axis,
                    start: self.start.clone() + &other.start,
                    end: self.end.clone() + &other.start,
                },
            )
            .map(Some);
        }
        Ok(None)
    }
}

/// Try to read `[lo, hi)` bounds on `axis_sym` out of an ROI of the canonical
/// indicator shape `Mul(Ge(axis, lo), Ge(hi-1, axis))`.
fn bounds_on_axis(roi: &TDim, axis_sym: &Symbol) -> Option<(i64, i64)> {
    let TDim::Mul(terms) = roi else { return None };
    if terms.len() != 2 {
        return None;
    }
    let mut lo: Option<i64> = None;
    let mut hi: Option<i64> = None;
    for term in terms {
        let TDim::Ge(left, right) = term else { return None };
        if let TDim::Sym(s) = left.as_ref()
            && s == axis_sym
            && let Ok(v) = right.to_i64()
        {
            lo = Some(v);
            continue;
        }
        if let TDim::Sym(s) = right.as_ref()
            && s == axis_sym
            && let Ok(v) = left.to_i64()
        {
            hi = Some(v + 1);
            continue;
        }
        return None;
    }
    Some((lo?, hi?))
}

/// Experimental TypedPass: walks the model, asks every op's `input_roi` for
/// single-axis band ROIs, and materialises each one by inserting
/// `Concat([zeros, Slice, zeros])` on the corresponding input.
///
/// **Currently NOT wired into the declutter pipeline** — confirmed to cause
/// an infinite loop with `PushSliceUp` (Mathieu's prior warning).  The
/// cycle-prevention check below (skip nodes that are part of a materialised
/// Concat-with-zero-Const structure) is inadequate: after a materialisation,
/// `PushSliceUp` pushes the inserted Slice further upstream, creating new
/// Concat/Slice structures that re-trigger materialisation.  Preserved as
/// `#[allow(dead_code)]` for the writeup discussion.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct MaterialiseBandRoi;

impl crate::optim::TypedPass for MaterialiseBandRoi {
    fn reset(&mut self) -> TractResult<()> {
        Ok(())
    }

    fn next(
        &mut self,
        _session: &mut crate::optim::OptimizerSession,
        model: &TypedModel,
    ) -> TractResult<Option<TypedModelPatch>> {
        let order = model.eval_order()?;
        for &nid in &order {
            let node = &model.nodes()[nid];

            // Cycle prevention: skip nodes that are PART of an already-
            // materialised structure (Concats with zero-Const segments, or
            // Slices we inserted).  PropagateRoi will keep re-tagging wires
            // inside the materialised chain; without this guard we'd
            // re-materialise indefinitely.
            if is_materialised_node(model, node)? {
                continue;
            }

            let Some(input_rois) = node.op.as_typed().unwrap().input_roi(model, node)? else {
                continue;
            };
            for (input_idx, roi) in input_rois.iter().enumerate() {
                let Some(roi) = roi else { continue };
                let in_fact = model.outlet_fact(node.inputs[input_idx])?;
                for axis in 0..in_fact.shape.rank() {
                    let axis_sym = model.symbols.coord_sym(axis);
                    let Some((lo, hi)) = bounds_on_axis(roi, &axis_sym) else { continue };
                    if let Some(patch) =
                        materialise_band_roi_on_input(model, node, input_idx, axis, lo, hi)?
                    {
                        return Ok(Some(patch));
                    }
                }
            }
        }
        Ok(None)
    }
}

/// Recognise a node that's part of an already-materialised band ROI
/// structure: a `TypedConcat` with at least one zero-Const input.  We also
/// recognise the inserted Slice itself (whose producer is in such a Concat).
#[allow(dead_code)]
fn is_materialised_node(model: &TypedModel, node: &TypedNode) -> TractResult<bool> {
    if node.op_as::<crate::ops::array::TypedConcat>().is_some() {
        for &inp in &node.inputs {
            let f = model.outlet_fact(inp)?;
            if f.konst.as_ref().is_some_and(|t| t.is_zero().unwrap_or(false)) {
                return Ok(true);
            }
        }
    }
    // The inserted Slice's sole consumer is the materialised Concat.
    if node.op_as::<Slice>().is_some() {
        for cons in model.outlet_successors(OutletId::new(node.id, 0)) {
            let cons_node = model.node(cons.node);
            if let Some(c) = cons_node.op_as::<crate::ops::array::TypedConcat>() {
                let _ = c;
                for &inp in &cons_node.inputs {
                    let f = model.outlet_fact(inp)?;
                    if f.konst.as_ref().is_some_and(|t| t.is_zero().unwrap_or(false)) {
                        return Ok(true);
                    }
                }
            }
        }
    }
    Ok(false)
}

/// Materialise a single-axis band `input_roi` for input `input_idx` of `node`
/// by wiring `Concat([zeros, Slice(input, axis, [lo, hi)), zeros], axis)` in
/// place of the original input.  The op is rewired against the modified
/// inputs.  Input shape is preserved by the Concat (so `shunt_outside`
/// accepts); the actual narrowing is the inserted Slice, which then bubbles
/// up via `PushSliceUp` through whatever produced the input.
///
/// Returns `None` if the input is already wrapped in a `Concat` with at least
/// one zero-Const segment (= we already materialised it; don't re-fire).
#[allow(dead_code)]
pub fn materialise_band_roi_on_input(
    model: &TypedModel,
    node: &TypedNode,
    input_idx: usize,
    axis: usize,
    lo: i64,
    hi: i64,
) -> TractResult<Option<TypedModelPatch>> {
    let input_outlet = node.inputs[input_idx];
    let input_fact = model.outlet_fact(input_outlet)?;
    let full_size = match input_fact.shape[axis].to_i64() {
        Ok(s) => s,
        Err(_) => return Ok(None),
    };
    let lo = lo.max(0);
    let hi = hi.min(full_size);
    if lo == 0 && hi == full_size {
        return Ok(None);
    }
    if lo >= hi {
        return Ok(None);
    }

    // Cycle prevention: skip if the input's producer is already a
    // TypedConcat with a zero-Const segment (= we already materialised).
    let producer = model.node(input_outlet.node);
    if let Some(_) = producer.op_as::<crate::ops::array::TypedConcat>() {
        for &inp in &producer.inputs {
            let f = model.outlet_fact(inp)?;
            if f.konst.as_ref().map(|t| t.is_zero().unwrap_or(false)).unwrap_or(false) {
                return Ok(None);
            }
        }
    }

    let mut patch =
        TypedModelPatch::new(format!("materialise_band_roi@{}.in{}", node.name, input_idx));

    // Tap each of N's inputs.  For input_idx, replace with Concat-wrapped Slice.
    let mut new_inputs: TVec<OutletId> = tvec!();
    let dt = input_fact.datum_type;
    let full_shape: TVec<usize> =
        input_fact.shape.iter().map(|d| d.to_usize()).collect::<TractResult<_>>()?;
    for (i, &inp) in node.inputs.iter().enumerate() {
        let tapped = patch.tap_model(model, inp)?;
        if i == input_idx {
            let narrow = patch.wire_node(
                format!("{}.in{}.narrow", node.name, i),
                Slice { axis, start: TDim::Val(lo), end: TDim::Val(hi) },
                &[tapped],
            )?[0];
            let mut concat_inputs: TVec<OutletId> = tvec!();
            if lo > 0 {
                let mut s = full_shape.clone();
                s[axis] = lo as usize;
                let z = Tensor::zero_dt(dt, &s)?.into_arc_tensor();
                concat_inputs
                    .push(patch.add_const(format!("{}.in{}.zeros_left", node.name, i), z)?);
            }
            concat_inputs.push(narrow);
            if hi < full_size {
                let mut s = full_shape.clone();
                s[axis] = (full_size - hi) as usize;
                let z = Tensor::zero_dt(dt, &s)?.into_arc_tensor();
                concat_inputs
                    .push(patch.add_const(format!("{}.in{}.zeros_right", node.name, i), z)?);
            }
            let restored = patch.wire_node(
                format!("{}.in{}.restored", node.name, i),
                crate::ops::array::TypedConcat::new(axis),
                &concat_inputs,
            )?[0];
            new_inputs.push(restored);
        } else {
            new_inputs.push(tapped);
        }
    }

    // Wire N (cloned op) with the modified inputs.
    let new_outputs = patch.wire_node(&node.name, node.op.clone(), &new_inputs)?;
    for (i, &outlet) in new_outputs.iter().enumerate() {
        patch.shunt_outside(model, OutletId::new(node.id, i), outlet)?;
    }
    patch.obliterate(node.id)?;
    Ok(Some(patch))
}

/// (Sketch — not currently used.  Generic helper preserved for the writeup
/// discussion of how the per-op `narrow_via_roi_concat` rewrite would
/// generalise beyond `Slice` itself to axes-preserving ops via
/// `axes_mapping`.  Slice has its own implementation above because its
/// `axes_mapping` declares the slice axis as "disconnected", which this
/// generic walker would otherwise reject.)
#[allow(dead_code)]
fn declutter_narrow_via_roi_concat<O: TypedOp + Clone>(
    op: &O,
    model: &TypedModel,
    node: &TypedNode,
) -> TractResult<Option<TypedModelPatch>> {
    let output_fact = model.outlet_fact(OutletId::new(node.id, 0))?;
    let Some(roi) = &output_fact.region_of_interest else { return Ok(None) };

    let axes_mapping = op.axes_mapping(
        &node.inputs.iter().map(|i| model.outlet_fact(*i)).collect::<TractResult<TVec<_>>>()?,
        &[output_fact],
    )?;

    // Search for a single output axis bounded by the ROI.
    for out_axis in 0..output_fact.shape.rank() {
        let axis_sym = model.symbols.coord_sym(out_axis);
        let Some((lo, hi)) = bounds_on_axis(roi, &axis_sym) else { continue };
        let out_dim_concrete = match output_fact.shape[out_axis].to_i64() {
            Ok(d) => d,
            Err(_) => continue,
        };
        if lo <= 0 && hi >= out_dim_concrete {
            continue; // Bounds don't narrow anything on this axis.
        }
        let lo = lo.max(0);
        let hi = hi.min(out_dim_concrete);

        // For each input, find the axis it contributes to this output axis.
        // For axes-preserving ops on this axis (in == out position), this is
        // straightforward; bail otherwise for the first sketch.
        let n_inputs = node.inputs.len();
        let mut input_axis_for: TVec<Option<usize>> = tvec!();
        for i in 0..n_inputs {
            let in_axis = axes_mapping.track_axis((InOut::Out(0), out_axis), InOut::In(i))?;
            input_axis_for.push(in_axis);
        }
        // Require ALL inputs to have a corresponding axis (= no contraction here).
        if input_axis_for.iter().any(|a| a.is_none()) {
            continue;
        }

        // Build the patch.
        let mut patch = TypedModelPatch::new(format!("narrow_via_roi@{}", node.name));
        let dt = output_fact.datum_type;
        let mut narrow_inputs: TVec<OutletId> = tvec!();
        for (i, &input) in node.inputs.iter().enumerate() {
            let in_axis = input_axis_for[i].unwrap();
            let tapped = patch.tap_model(model, input)?;
            let in_fact = model.outlet_fact(input)?;
            let in_dim = match in_fact.shape[in_axis].to_i64() {
                Ok(d) => d,
                Err(_) => return Ok(None),
            };
            // For this sketch, only handle the case where the input axis size
            // equals the output axis size (no broadcasting / no offset).  This
            // covers axes-preserving ops on the band axis.
            if in_dim != out_dim_concrete {
                return Ok(None);
            }
            let sliced = patch.wire_node(
                format!("{}.narrow_in{i}", node.name),
                Slice { axis: in_axis, start: TDim::Val(lo), end: TDim::Val(hi) },
                &[tapped],
            )?[0];
            narrow_inputs.push(sliced);
        }

        // Apply the original op to the narrowed inputs.
        let narrow_op = op.clone();
        let narrow_out =
            patch.wire_node(format!("{}.narrow", node.name), narrow_op, &narrow_inputs)?[0];

        // Build zero pads on the band axis.
        let mut left_shape: TVec<usize> =
            output_fact.shape.iter().map(|d| d.to_usize()).collect::<TractResult<_>>()?;
        left_shape[out_axis] = lo as usize;
        let mut right_shape = left_shape.clone();
        right_shape[out_axis] = (out_dim_concrete - hi) as usize;

        let mut concat_inputs: TVec<OutletId> = tvec!();
        if lo > 0 {
            let zeros_left = Tensor::zero_dt(dt, &left_shape)?.into_arc_tensor();
            let z = patch.add_const(format!("{}.narrow_zeros_left", node.name), zeros_left)?;
            concat_inputs.push(z);
        }
        concat_inputs.push(narrow_out);
        if hi < out_dim_concrete {
            let zeros_right = Tensor::zero_dt(dt, &right_shape)?.into_arc_tensor();
            let z = patch.add_const(format!("{}.narrow_zeros_right", node.name), zeros_right)?;
            concat_inputs.push(z);
        }

        let concat_out = patch.wire_node(
            format!("{}.narrow_concat", node.name),
            crate::ops::array::TypedConcat::new(out_axis),
            &concat_inputs,
        )?[0];

        patch.shunt_outside(model, OutletId::new(node.id, 0), concat_out)?;
        return Ok(Some(patch));
    }
    Ok(None)
}

impl Op for Slice {
    fn name(&self) -> StaticName {
        "Slice".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axis: {}, {}..{}", self.axis, self.start, self.end)])
    }

    op_as_typed_op!();
}

impl EvalOp for Slice {
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
        let start = self.start.eval(&session.resolved_symbols).to_usize()?;
        let end = self.end.eval(&session.resolved_symbols).to_usize()?;
        eval_slice(&input, self.axis, start, end)
    }
}

fn eval_slice(input: &Tensor, axis: usize, start: usize, end: usize) -> TractResult<TVec<TValue>> {
    if end > input.shape()[axis] || start > end {
        bail!("Invalid range {}..{} for slicing {:?} on axis {}", start, end, input, axis);
    }
    unsafe {
        let mut shape: TVec<_> = input.shape().into();
        shape[axis] = end - start;
        let mut tensor = Tensor::uninitialized_dt(input.datum_type(), &shape)?;
        tensor.assign_slice_unchecked(.., input, start..end, axis);
        Ok(tvec!(tensor.into_tvalue()))
    }
}

impl TypedOp for Slice {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        anyhow::ensure!(inputs.len() == 1, "Slice has one single input");
        if let (Ok(start), Ok(end), Ok(len)) =
            (self.start.to_usize(), self.end.to_usize(), inputs[0].shape[self.axis].to_usize())
        {
            ensure!(start <= end);
            ensure!(end <= len);
        }
        let mut fact = inputs[0].without_value();
        fact.shape.set(self.axis, (self.end.clone() - &self.start).to_dim());
        Ok(tvec!(fact))
    }

    fn input_roi(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TVec<Option<TDim>>>> {
        let output_fact = model.outlet_fact(OutletId::new(node.id, 0))?;
        rule_if_some!(roi = &output_fact.region_of_interest);
        if self.start.is_zero() {
            return Ok(Some(tvec![Some(roi.clone())]));
        }
        // Remap: output 🎯axis = input 🎯axis - start, so substitute 🎯axis → 🎯axis + start
        if let Some(sym) = roi
            .symbols()
            .into_iter()
            .find(|s| crate::ops::logic::sym_to_coord_axis(s) == Some(self.axis))
        {
            let shifted = TDim::Sym(sym.clone()) + self.start.clone();
            if let Ok(input_roi) = roi.substitute(&sym, &shifted) {
                return Ok(Some(tvec![Some(input_roi)]));
            }
        }
        // ROI doesn't mention the sliced axis — pass through unchanged
        Ok(Some(tvec![Some(roi.clone())]))
    }

    fn axes_mapping(
        &self,
        inputs: &[&TypedFact],
        outputs: &[&TypedFact],
    ) -> TractResult<AxesMapping> {
        let mut mapping = AxesMapping::disconnected(inputs, outputs)?;
        for (axis, repr) in (0..inputs[0].rank()).zip('a'..) {
            if self.axis != axis {
                mapping = mapping
                    .renaming((InOut::In(0), axis), repr)?
                    .linking(repr, (InOut::Out(0), axis))?;
            }
        }
        Ok(mapping)
    }

    fn change_axes(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        _io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        if let Some(axis) = change.transform_axis(self.axis) {
            if axis != self.axis {
                Ok(Some(AxisChangeConsequence::new(
                    model,
                    node,
                    Some(Box::new(Slice { axis, ..self.clone() }) as _),
                    change,
                )))
            } else {
                Ok(Some(AxisChangeConsequence::new(model, node, None, change)))
            }
        } else {
            Ok(None)
        }
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if self.start.is_zero() && (self.end == model.outlet_fact(node.inputs[0])?.shape[self.axis])
        {
            TypedModelPatch::shunt_one_op(model, node)
        } else if let Some(p) = self.declutter_slice_after_slice(model, node)? {
            Ok(Some(p))
        } else if let Some(p) = self.declutter_narrow_via_roi_concat(model, node)? {
            Ok(Some(p))
        } else {
            Ok(None)
        }
    }

    fn substitute_symbols(
        &self,
        _source: &TypedModel,
        node: &TypedNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
        subs: &HashMap<Symbol, TDim>,
    ) -> TractResult<TVec<OutletId>> {
        let op = Slice {
            axis: self.axis,
            start: self.start.substitute_all(subs)?,
            end: self.end.substitute_all(subs)?,
        };
        let inputs = node.inputs.iter().map(|i| mapping[i]).collect::<TVec<_>>();
        target.wire_node(&node.name, op, &inputs)
    }

    fn slice(
        &self,
        patch: &mut TypedModelPatch,
        _model: &TypedModel,
        node: &TypedNode,
        _prefix: &str,
        inputs: &[OutletId],
        _output_axis: usize,
        _start: &TDim,
        _end: &TDim,
    ) -> TractResult<Option<TVec<OutletId>>> {
        patch.wire_node(&node.name, &node.op, inputs).map(Some)
    }

    as_op!();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::array::TypedConcat;

    fn band_roi_on_axis_zero(scope: &SymbolScope, lo: i64, hi_inclusive: i64) -> TDim {
        let sym = scope.coord_sym(0);
        TDim::Mul(vec![
            TDim::Ge(Box::new(TDim::Sym(sym.clone())), Box::new(TDim::Val(lo))),
            TDim::Ge(Box::new(TDim::Val(hi_inclusive)), Box::new(TDim::Sym(sym))),
        ])
    }

    /// Single Slice with band ROI on its output: rewrite should produce
    /// `Concat(zeros, narrow_slice, zeros)`, same output shape.  Then the
    /// outer noop-slice should clean up; the standard Concat declutter should
    /// NOT eliminate the zero pads (they're not zero-volume).  End state: a
    /// 3-input Concat with the narrowed slice in the middle.
    #[test]
    fn roi_narrow_single_slice_produces_concat() -> TractResult<()> {
        let mut model = TypedModel::default();
        let src = model.add_source("src", f32::fact(&[100]))?;
        let sliced = model.wire_node("sliced", Slice::new(0, 2.to_dim(), 20.to_dim()), &[src])?[0];
        model.select_output_outlets(&[sliced])?;

        // Output ROI: band [3, 12) on axis 0 (3 ≤ 🎯0 ≤ 11).
        let roi = band_roi_on_axis_zero(&model.symbols, 3, 11);
        let sliced_node = sliced.node;
        model.nodes_mut()[sliced_node].outputs[0].fact.region_of_interest = Some(roi);

        let model = model.into_decluttered()?;

        // The output should still be width 18 (= end - start = 20 - 2).
        let out_fact = model.outlet_fact(model.output_outlets()?[0])?;
        assert_eq!(out_fact.shape[0].to_i64()?, 18);

        // We expect one Concat (the shape-preserver) wrapping a narrowed Slice.
        let concat_count = model.nodes().iter().filter(|n| n.op_is::<TypedConcat>()).count();
        let slice_nodes: Vec<&Slice> =
            model.nodes().iter().filter_map(|n| n.op_as::<Slice>()).collect();
        eprintln!(
            "post-declutter: {} Concat nodes, {} Slice nodes",
            concat_count,
            slice_nodes.len()
        );
        for s in &slice_nodes {
            eprintln!("  Slice axis={} start={} end={}", s.axis, s.start, s.end);
        }
        Ok(())
    }

    /// ROI bubbling via `PropagateRoi`: place band ROI on a downstream-of-Slice
    /// `AxisOp::Add` (passthrough on the slice axis), confirm it bubbles upstream
    /// to the Slice and triggers narrow-via-Concat there.
    #[test]
    fn roi_narrow_through_passthrough_op() -> TractResult<()> {
        use crate::ops::change_axes::AxisOp;
        let mut model = TypedModel::default();
        let src = model.add_source("src", f32::fact(&[100]))?;
        let sliced = model.wire_node("sliced", Slice::new(0, 2.to_dim(), 20.to_dim()), &[src])?[0];
        // Insert AxisOp::Add (passthrough on axis 0) — output rank 2: [1, 18].
        let added = model.wire_node("added", AxisOp::Add(0), &[sliced])?[0];
        model.select_output_outlets(&[added])?;

        // Put band ROI on the AxisOp's output, axis 1 (= the original slice axis,
        // shifted by 1 due to the AxisOp::Add(0)).  This should bubble via
        // PropagateRoi to the Slice's output (axis 0) and trigger narrow-via-Concat.
        let sym = model.symbols.coord_sym(1);
        let roi = TDim::Mul(vec![
            TDim::Ge(Box::new(TDim::Sym(sym.clone())), Box::new(TDim::Val(3))),
            TDim::Ge(Box::new(TDim::Val(11)), Box::new(TDim::Sym(sym))),
        ]);
        let added_node = added.node;
        model.nodes_mut()[added_node].outputs[0].fact.region_of_interest = Some(roi);

        let model = model.into_decluttered()?;

        eprintln!("=== final graph ===");
        for n in model.nodes() {
            eprintln!("#{} {} {}", n.id, n.op.name(), n.name);
            for (i, o) in n.outputs.iter().enumerate() {
                eprintln!("  out[{}]: {:?}", i, o.fact);
            }
        }
        Ok(())
    }

    /// **Theory validation**: an op (here `Iff`) whose `input_roi` reads a
    /// single-axis band from its `cond`'s `uniform_tdim` should — once
    /// `PropagateRoi` runs — produce a band ROI on the upstream `Slice`'s
    /// output, triggering `Slice::declutter_narrow_via_roi_concat` and
    /// narrowing the Slice via Concat-zeros.  No new mechanism needed on
    /// Iff's side: PropagateRoi handles the bubbling, my Slice declutter
    /// handles the materialization.
    #[test]
    fn iff_with_single_axis_band_cond_narrows_upstream_slice() -> TractResult<()> {
        use crate::ops::logic::Iff;

        let mut model = TypedModel::default();
        // Source: a [100] f32 tensor.
        let src = model.add_source("src", f32::fact(&[100]))?;
        // Slice the source: [100] → [18] (start=2, end=20).
        let sliced = model.wire_node("sliced", Slice::new(0, 2.to_dim(), 20.to_dim()), &[src])?[0];

        // Cond wire: a bool [18] with single-axis band uniform_tdim baked
        // into the TypedSource's fact (so `output_facts` doesn't clobber it).
        let axis_sym = model.symbols.coord_sym(0);
        let cond_utdim = TDim::Mul(vec![
            TDim::Ge(Box::new(TDim::Sym(axis_sym.clone())), Box::new(TDim::Val(3))),
            TDim::Ge(Box::new(TDim::Val(11)), Box::new(TDim::Sym(axis_sym))),
        ]);
        let mut cond_fact = bool::fact(&[18]);
        cond_fact.uniform_tdim = Some(cond_utdim);
        let cond_src = model.add_source("cond", cond_fact)?;

        // Else branch: zero const, shape [18].
        let else_branch = model.add_source("else_b", f32::fact(&[18]))?;

        // Iff(cond, sliced, else_branch).  Iff's `input_roi` will read
        // cond's uniform_tdim and plant it as the ROI on `sliced`.
        let iff = model.wire_node("iff", Iff, &[cond_src, sliced, else_branch])?[0];
        model.select_output_outlets(&[iff])?;

        let model = model.into_decluttered()?;

        eprintln!("=== iff cascade test, final graph ===");
        for n in model.nodes() {
            eprintln!("#{} {} {}", n.id, n.op.name(), n.name);
            for (i, o) in n.outputs.iter().enumerate() {
                eprintln!("  out[{}]: {:?}", i, o.fact);
            }
        }

        // We expect to see a narrowed Slice somewhere (start=5 OR end=14)
        // — the band [3, 12) of an [18]-wide output translates to input
        // coordinates [start + 3, start + 12) = [5, 14) of the [100] source.
        let slice_nodes: Vec<&Slice> =
            model.nodes().iter().filter_map(|n| n.op_as::<Slice>()).collect();
        let has_narrowed =
            slice_nodes.iter().any(|s| s.start == 5.to_dim() && s.end == 14.to_dim());
        assert!(
            has_narrowed,
            "expected a Slice(5, 14) somewhere in the post-declutter graph; got {:?}",
            slice_nodes.iter().map(|s| (s.start.clone(), s.end.clone())).collect::<Vec<_>>()
        );
        Ok(())
    }

    /// **Theory validation, multi-op chain**: `Source → Slice(2, 20) →
    /// AxisOp::Add(0) → Iff(banded_cond, [Add output], else)`.  The chain
    /// between the Slice and Iff is non-Slice (AxisOp::Add); for the
    /// cascade to materialize the narrowing all the way to the source-side
    /// Slice, the ROI on the Add output (post-bubble) must trigger a
    /// Slice INSERTION on that wire, which `PushSliceUp` then pushes
    /// upstream through the Add, eventually reaching the existing Slice
    /// where Slice-after-Slice composes them.
    ///
    /// This test pins what's currently missing: the materialization step
    /// for *non-Slice* wires.  Today the ROI bubbles to the Slice's output
    /// via PropagateRoi, but the experiment validates the Slice-side path
    /// works.  Marked `#[ignore]` until the generic-materialization
    /// mechanism lands.
    #[test]
    #[ignore = "MaterialiseBandRoi pass causes infinite loop with PushSliceUp; \
                cycle prevention not yet adequate (Mathieu's prior warning confirmed)"]
    fn iff_with_axisop_between_slice_and_iff() -> TractResult<()> {
        use crate::ops::change_axes::AxisOp;
        use crate::ops::logic::Iff;

        let mut model = TypedModel::default();
        let src = model.add_source("src", f32::fact(&[100]))?;
        let sliced = model.wire_node("sliced", Slice::new(0, 2.to_dim(), 20.to_dim()), &[src])?[0];
        let added = model.wire_node("added", AxisOp::Add(0), &[sliced])?[0];

        let axis_sym = model.symbols.coord_sym(1);
        let cond_utdim = TDim::Mul(vec![
            TDim::Ge(Box::new(TDim::Sym(axis_sym.clone())), Box::new(TDim::Val(3))),
            TDim::Ge(Box::new(TDim::Val(11)), Box::new(TDim::Sym(axis_sym))),
        ]);
        let mut cond_fact = bool::fact(&[1, 18]);
        cond_fact.uniform_tdim = Some(cond_utdim);
        let cond_src = model.add_source("cond", cond_fact)?;

        let else_branch = model.add_source("else_b", f32::fact(&[1, 18]))?;
        let iff = model.wire_node("iff", Iff, &[cond_src, added, else_branch])?[0];
        model.select_output_outlets(&[iff])?;

        let model = model.into_decluttered()?;

        let slice_nodes: Vec<&Slice> =
            model.nodes().iter().filter_map(|n| n.op_as::<Slice>()).collect();
        let has_narrowed =
            slice_nodes.iter().any(|s| s.start == 5.to_dim() && s.end == 14.to_dim());
        assert!(
            has_narrowed,
            "expected a Slice(5, 14) somewhere; got {:?}",
            slice_nodes.iter().map(|s| (s.start.clone(), s.end.clone())).collect::<Vec<_>>()
        );
        Ok(())
    }

    /// Cascade test: `Source → Slice(2,20) → Slice(0,18)` with band ROI on the
    /// downstream slice.  Expect declutter+PropagateRoi cascade to collapse the
    /// chain to a single narrowed slice.
    #[test]
    fn roi_narrow_cascade_two_slices_collapses() -> TractResult<()> {
        let mut model = TypedModel::default();
        let src = model.add_source("src", f32::fact(&[100]))?;
        let s1 = model.wire_node("s1", Slice::new(0, 2.to_dim(), 20.to_dim()), &[src])?[0];
        let s2 = model.wire_node("s2", Slice::new(0, 0.to_dim(), 18.to_dim()), &[s1])?[0];
        model.select_output_outlets(&[s2])?;

        let roi = band_roi_on_axis_zero(&model.symbols, 3, 11);
        let s2_node = s2.node;
        model.nodes_mut()[s2_node].outputs[0].fact.region_of_interest = Some(roi);

        let model = model.into_decluttered()?;

        let out_fact = model.outlet_fact(model.output_outlets()?[0])?;
        eprintln!("Final output fact: {:?}", out_fact);
        let slice_nodes: Vec<&Slice> =
            model.nodes().iter().filter_map(|n| n.op_as::<Slice>()).collect();
        let concat_count = model.nodes().iter().filter(|n| n.op_is::<TypedConcat>()).count();
        eprintln!("post-declutter: {} Concat, {} Slice", concat_count, slice_nodes.len());
        for s in &slice_nodes {
            eprintln!("  Slice axis={} start={} end={}", s.axis, s.start, s.end);
        }
        Ok(())
    }
}
