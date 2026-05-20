use std::borrow::Borrow;
use std::fmt::Debug;

use crate::internal::*;
use crate::ops::array::MultiBroadcastTo;
use crate::tract_data::itertools::Itertools;

mod eval;

pub mod einsum_matmul;
pub mod kernel_selection;
pub mod prefix_matmul;

#[cfg(test)]
mod proptest;

use num_traits::One;
use tract_linalg::block_quant::{BlockQuantFact, PackedBlockQuantFact};
use tract_linalg::mmm::PackedExoticFact;

pub fn block_quant_aware_input_shape(fact: &TypedFact) -> TractResult<Cow<'_, [TDim]>> {
    if fact.is_plain() {
        return Ok(Cow::Borrowed(&*fact.shape));
    }
    let Some(exotic_fact) = fact.exotic_fact() else {
        bail!("Datum fact is exotic, but no exotic fact was found.")
    };
    if let Some(_bqf) = exotic_fact.downcast_ref::<BlockQuantFact>() {
        Ok(Cow::Borrowed(&*fact.shape))
    } else if let Some(pof) = exotic_fact.downcast_ref::<PackedBlockQuantFact>() {
        Ok(Cow::Owned(
            fact.shape.iter().cloned().chain(pof.shape.iter().map(|i| i.to_dim())).collect_vec(),
        ))
    } else if let Some(pof) = exotic_fact.downcast_ref::<PackedExoticFact>() {
        Ok(Cow::Owned(
            fact.shape.iter().cloned().chain([pof.mn.clone(), pof.k.to_dim()]).collect_vec(),
        ))
    } else {
        bail!("Unsupported exotic fact {exotic_fact:?}")
    }
}

#[derive(Clone, Hash, PartialEq, Eq)]
pub struct EinSum {
    pub axes: AxesMapping,
    pub operating_dt: DatumType,
    // if present, assume we're a binary op.
    // 9 inputs are: A,B,bias, A0,Ascale, B0,BScale, C0,Cscale
    pub q_params: Option<DatumType>,
}

impl EinSum {
    pub fn new(axes: AxesMapping, operating_dt: DatumType) -> EinSum {
        EinSum { axes, operating_dt, q_params: None }
    }

    pub fn newq(axes: AxesMapping, operating_dt: DatumType, output_type: DatumType) -> EinSum {
        EinSum { axes, operating_dt, q_params: Some(output_type) }
    }

    pub fn actual_input_shapes_from_facts<'m>(
        &self,
        inputs: &'m [impl Borrow<TypedFact>],
    ) -> TractResult<TVec<Cow<'m, [TDim]>>> {
        ensure!(inputs.len() == self.axes.input_count());
        let shapes: TVec<Cow<[TDim]>> = inputs
            .iter()
            .map(|t| block_quant_aware_input_shape(t.borrow()))
            .collect::<TractResult<_>>()?;
        ensure!(
            shapes.iter().enumerate().all(|(ix, fact)| fact.len() == self.axes.rank(InOut::In(ix)))
        );
        Ok(shapes)
    }

    #[allow(unused_variables)]
    pub(crate) fn propagate_axis(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        io: InOut,
        axis: usize,
    ) -> TractResult<Option<TypedModelPatch>> {
        let mut new_axis = self.axes.axis((io, axis))?.clone();
        let repr = new_axis.repr;
        let mut patch = TypedModelPatch::new(format!("Propagate axis {}", new_axis.repr));
        let mut taps = tvec!();
        for (ix, input) in node.inputs.iter().enumerate() {
            let mut tap = patch.tap_model(model, *input)?;
            rule_if!(new_axis.inputs[ix].len() <= 1); // FIXME maybe
            if new_axis.inputs[ix].is_empty() {
                let insert_at = self.axes.rank(InOut::In(ix));
                tap = patch.wire_node(
                    format!("{}.prop_axis.{}.input_{}", &node.name, new_axis.repr, ix),
                    AxisOp::Add(insert_at),
                    &[tap],
                )?[0];
                new_axis.inputs[ix].push(insert_at);
            }
            taps.push(tap);
        }
        let must_rm_axis: Option<usize> = if new_axis.outputs[0].len() == 0 {
            let insert_at = self.axes.rank(InOut::Out(0));
            new_axis.outputs[0].push(insert_at);
            Some(insert_at)
        } else {
            None
        };
        let new_expr = self
            .axes
            .iter_all_axes()
            .map(|it| if it.repr == new_axis.repr { new_axis.clone() } else { it.clone() })
            .collect_vec();
        let axes = AxesMapping::new(node.inputs.len(), 1, new_expr)?;
        let mut wire = patch.wire_node(&node.name, Self { axes, ..self.clone() }, &taps)?;
        if let Some(position) = must_rm_axis {
            wire = patch.wire_node(
                format!("{}.prop_axis.{}.output", &node.name, repr),
                AxisOp::Rm(position),
                &wire,
            )?;
        }
        patch.shunt_outside(model, node.id.into(), wire[0])?;
        Ok(Some(patch))
    }

    pub fn acceptable_accumulators(&self) -> TVec<DatumType> {
        if self.operating_dt.is_integer() {
            tvec!(i32::datum_type())
        } else if self.operating_dt == f16::datum_type() {
            tvec!(f16::datum_type(), f32::datum_type())
        } else {
            tvec!(self.operating_dt)
        }
    }
}

impl Debug for EinSum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "EinSum {} ({:?})", self.axes, self.operating_dt)
    }
}

impl Op for EinSum {
    fn name(&self) -> StaticName {
        "EinSum".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        let mut info = vec![format!("{} ({:?})", self.axes, self.operating_dt)];
        if let Some(qp) = self.q_params {
            info.push(format!("Quantized output: {qp:?}"));
        }
        Ok(info)
    }

    op_as_typed_op!();
}

impl EvalOp for EinSum {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        if inputs.iter().all(|i| i.datum_type().is_number() && i.is_plain()) {
            let mut adhoc_model = TypedModel::default();
            let mut wires = tvec!();
            for (ix, input) in inputs.iter().enumerate() {
                let fact = TypedFact::shape_and_dt_of(input);
                let wire = adhoc_model.add_source(format!("input.{ix}"), fact)?;
                wires.push(wire);
            }
            let output = adhoc_model.wire_node("einsum", self.clone(), &wires)?;
            adhoc_model.select_output_outlets(&output)?;
            let opti = adhoc_model.into_optimized()?;
            if opti.nodes.iter().all(|node| !node.op_is::<Self>()) {
                return opti.into_runnable()?.run(inputs);
            }
        }

        let output = if let Some(qp) = self.q_params {
            eval::eval_q(&self.axes, qp, inputs)
        } else {
            dispatch_numbers!(eval::eval_t(self.operating_dt)(&self.axes, inputs))
        }?;
        Ok(tvec!(output.into_tvalue()))
    }
}

impl TypedOp for EinSum {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let shapes = self.actual_input_shapes_from_facts(inputs)?;
        for i in 0..inputs.len() {
            ensure!(shapes[i].len() == self.axes.rank(InOut::In(i)));
        }
        for axis in self.axes.iter_all_axes() {
            assert!(
                shapes
                    .iter()
                    .enumerate()
                    .flat_map(|(slot, shape)| axis.inputs[slot].iter().map(|a| &shape[*a]))
                    .try_fold(TDim::one(), |a, b| TDim::broadcast(a, b.clone()))
                    .is_ok()
            );
        }
        if let Some(qp) = self.q_params {
            ensure!(inputs.len() == 9);
            Ok(tvec!(qp.fact(eval::output_shape(&self.axes, &shapes[0..2])?)))
        } else {
            Ok(tvec!(TypedFact::dt_shape(
                self.operating_dt,
                eval::output_shape(&self.axes, &shapes)?
            )))
        }
    }

    fn input_roi(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TVec<Option<TDim>>>> {
        // First try bubble_roi: works for inputs that cover all ROI coord
        // axes mentioned in the output ROI.  For inputs that DON'T cover
        // every coord axis (= contracted/projected-out axes from this
        // input's perspective), try the closed-form chunked-band recogniser
        // which yields a constant band on the input's kept axis after
        // existentially quantifying the projected axes.
        let output_fact = model.outlet_fact(OutletId::new(node.id, 0))?;
        let Some(roi) = &output_fact.region_of_interest else { return Ok(None) };
        let input_facts: TVec<&TypedFact> =
            node.inputs.iter().map(|i| model.outlet_fact(*i)).collect::<TractResult<_>>()?;
        let output_facts = tvec![output_fact];
        let inputs_ref: Vec<&TypedFact> = input_facts.iter().copied().collect();
        let outputs_ref: Vec<&TypedFact> = output_facts.iter().copied().collect();
        let mapping = self.axes_mapping(&inputs_ref, &outputs_ref)?;
        let roi_coord_axes: Vec<(usize, Symbol)> = roi
            .symbols()
            .into_iter()
            .filter_map(|s| crate::ops::logic::sym_to_coord_axis(&s).map(|k| (k, s)))
            .collect();

        let project_for_input = |input_ix: usize| -> Option<TDim> {
            // Classify each output ROI coord axis: projected (no input axis)
            // or preserved (maps to input).
            let mut projected: Vec<Symbol> = vec![];
            let mut preserved: Vec<(Symbol, usize)> = vec![];
            for (out_pos, sym) in &roi_coord_axes {
                let logical = mapping
                    .iter_all_axes()
                    .find(|a| a.outputs.first().is_some_and(|o| o.contains(out_pos)))?;
                match logical.inputs[input_ix].first() {
                    None => projected.push(sym.clone()),
                    Some(&in_pos) => {
                        if input_facts[input_ix].shape[in_pos] != output_fact.shape[*out_pos] {
                            return None;
                        }
                        preserved.push((sym.clone(), in_pos));
                    }
                }
            }
            if projected.is_empty() {
                // All axes preserved — fall through to standard remap.
                let mut sub_map: HashMap<Symbol, TDim> = HashMap::new();
                for (sym, in_pos) in &preserved {
                    if crate::ops::logic::sym_to_coord_axis(sym) != Some(*in_pos) {
                        let scope = sym.scope()?;
                        sub_map.insert(sym.clone(), TDim::Sym(scope.coord_sym(*in_pos)));
                    }
                }
                return if sub_map.is_empty() {
                    Some(roi.clone())
                } else {
                    roi.substitute_all(&sub_map).ok()
                };
            }
            // Try the chunked-band recogniser: one projected axis × one
            // preserved axis at a time.
            for p_sym in &projected {
                for (k_sym, k_in_pos) in &preserved {
                    if let Some(band) = crate::optim::propagate_roi::recognise_chunked_band_project(
                        roi, p_sym, k_sym,
                    ) {
                        // Result mentions k_sym (output frame).  Remap to
                        // input axis position.
                        if crate::ops::logic::sym_to_coord_axis(k_sym) != Some(*k_in_pos) {
                            let scope = k_sym.scope()?;
                            let mut m: HashMap<Symbol, TDim> = HashMap::new();
                            m.insert(k_sym.clone(), TDim::Sym(scope.coord_sym(*k_in_pos)));
                            return band.substitute_all(&m).ok();
                        }
                        return Some(band);
                    }
                }
            }
            None
        };
        let result: TVec<Option<TDim>> =
            (0..node.inputs.len()).map(|ix| project_for_input(ix)).collect();
        Ok(Some(result))
    }

    fn axes_mapping(
        &self,
        inputs: &[&TypedFact],
        _outputs: &[&TypedFact],
    ) -> TractResult<AxesMapping> {
        let mut axes = self.axes.clone();
        for (slot, i) in inputs.iter().enumerate() {
            if i.is_exotic()
                && (i.exotic_fact().is_some_and(|of| {
                    of.is::<PackedExoticFact>() || of.is::<PackedBlockQuantFact>()
                }))
            {
                axes = axes
                    .remove_axis_occurency(InOut::In(slot), i.rank())?
                    .remove_axis_occurency(InOut::In(slot), i.rank())?;
            }
        }
        Ok(axes)
    }

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        let shapes = self.actual_input_shapes_from_facts(inputs)?;
        let oshape = eval::output_shape(&self.axes, &shapes)?;
        let ks = self
            .axes
            .iter_all_axes()
            .filter(|axis| axis.outputs[0].len() == 0)
            .map(|axis| {
                axis.inputs
                    .iter()
                    .enumerate()
                    .flat_map(|(ix, axes)| {
                        axes.iter()
                            .map(|axis| shapes[ix][*axis].clone())
                            .collect::<TVec<_>>()
                            .into_iter()
                    })
                    .find(|d| !d.is_one())
                    .unwrap_or_else(|| 1.to_dim())
            })
            .product::<TDim>();
        Ok(tvec!((Cost::FMA(self.operating_dt), oshape.iter().product::<TDim>() * ks)))
    }

    fn slice(
        &self,
        patch: &mut TypedModelPatch,
        model: &TypedModel,
        node: &TypedNode,
        prefix: &str,
        inputs: &[OutletId],
        output_axis: usize,
        _start: &TDim,
        _end: &TDim,
    ) -> TractResult<Option<TVec<OutletId>>> {
        let facts = model.node_input_facts(node.id)?;
        let axis = self.axes.axis((InOut::Out(0), output_axis))?;
        if facts
            .iter()
            .enumerate()
            .any(|(slot, fact)| axis.inputs[slot].len() > 0 && fact.is_exotic())
        {
            Ok(None)
        } else {
            patch.wire_node(prefix, self.clone(), inputs).map(Some)
        }
    }

    #[allow(unused_variables)]
    fn change_axes(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        let (mut inputs, mut outputs) = self.axes.to_strs();
        let interface: &mut String = match io {
            InOut::In(i) => &mut inputs[i],
            InOut::Out(o) => &mut outputs[o],
        };
        let mut axes: Vec<char> = interface.chars().collect();
        match change {
            AxisOp::Rm(rm) => {
                axes.remove(*rm);
            }
            AxisOp::Add(add) => axes.insert(*add, self.axes.available_label()),
            AxisOp::Move(from, to) => {
                let c = axes.remove(*from);
                axes.insert(*to, c);
            }
            _ => {
                return Ok(None);
            }
        };
        *interface = axes.into_iter().collect();
        let axes = AxesMapping::from_strs(&inputs, &outputs)?;
        Ok(Some(AxisChangeConsequence {
            substitute_op: Some(Box::new(EinSum { axes, ..self.clone() })),
            wire_changes: tvec!((io, change.clone())),
        }))
    }

    fn declutter_with_session(
        &self,
        session: &mut crate::optim::OptimizerSession,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if let Some(patch) = declutter_reshape_folding_input_axis(self, session, model, node)? {
            return Ok(Some(patch));
        }
        if let Some(patch) = declutter_broadcast(self, session, model, node)? {
            return Ok(Some(patch));
        }
        if let Some(patch) = unit_k_to_broadcast_mul(self, model, node)? {
            return Ok(Some(patch));
        }
        Ok(None)
    }

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        rule_if!(
            (self.q_params.is_none() && node.inputs.len() == 2)
                || (self.q_params.is_some() && node.inputs.len() == 9)
        );
        // Some EinSums are introduced during codegen itself (e.g. ConvTranspose lowering
        // emits an EinSum + DeconvSum pair). Those don't get a chance to go through declutter
        // before being lowered, so we re-check the unit-K → broadcast-Mul rule here as a
        // fast path. For EinSums that already existed at declutter time, this is a no-op
        // (the declutter pass would already have rewritten them).
        if let Some(patch) = unit_k_to_broadcast_mul(self, model, node)? {
            return Ok(Some(patch));
        }
        einsum_matmul::detect_rule(&(), model, node, &node.name, self)
    }

    as_op!();
}

fn declutter_reshape_folding_input_axis(
    op: &EinSum,
    _session: &mut crate::optim::OptimizerSession,
    model: &TypedModel,
    node: &TypedNode,
) -> TractResult<Option<TypedModelPatch>> {
    for (slot, prec) in node.inputs.iter().map(|n| model.node(n.node)).enumerate() {
        let Some(&AxisOp::Reshape(at, ref from, ref to)) = prec.op_as() else { continue };
        if to.len() > 1 {
            continue;
        }
        let mut axes = op.axes.clone();
        let extra_labels = axes.available_labels().take(from.len() - 1).collect_vec();
        // add a temporary input to axes to hold the extra axes
        let extra_input = node.inputs.len();
        axes = axes.with_extra_input(extra_input)?;
        for label in &extra_labels {
            axes = axes.with_extra_axis(*label, InOut::In(extra_input), 0)?;
        }
        let folded_axis = op.axes.axis((InOut::In(slot), at))?;
        rule_if!(folded_axis.outputs[0].len() <= 1);
        let mut patch = TypedModelPatch::default();
        let mut taps = patch.taps(model, &node.inputs)?;
        for (input, tap) in taps.iter_mut().enumerate() {
            if folded_axis.inputs[input].len() == 0 {
                continue;
            };
            rule_if!(folded_axis.inputs[input].len() <= 1);
            let pos = folded_axis.inputs[input][0];
            for label in &extra_labels {
                axes = axes.with_extra_axis_occurency(*label, InOut::In(input), pos)?;
            }
            *tap = patch.wire_node(
                format!("{}.reshape_folded_input_{}", node.name, input),
                AxisOp::Reshape(pos, to.clone(), from.clone()),
                &[*tap],
            )?[0];
        }
        if folded_axis.outputs[0].len() == 1 {
            let pos = folded_axis.outputs[0][0];
            for label in &extra_labels {
                axes = axes.with_extra_axis_occurency(*label, InOut::Out(0), pos)?;
            }
        }
        axes = axes.remove_slot(InOut::In(extra_input))?;
        let mut wire = patch.wire_node(&node.name, EinSum { axes, ..op.clone() }, &taps)?;
        if folded_axis.outputs[0].len() == 1 {
            let pos = folded_axis.outputs[0][0];
            wire = patch.wire_node(
                format!("{}.reshape_folded_output", node.name),
                AxisOp::Reshape(pos, from.clone(), to.clone()),
                &wire,
            )?;
        }
        patch.shunt_outside(model, node.id.into(), wire[0])?;
        return Ok(Some(patch));
    }
    Ok(None)
}

fn declutter_broadcast(
    op: &EinSum,
    _session: &mut crate::optim::OptimizerSession,
    model: &TypedModel,
    node: &TypedNode,
) -> TractResult<Option<TypedModelPatch>> {
    for (ix, outlet) in node.inputs.iter().enumerate() {
        let prec = model.node(outlet.node);
        if prec.op_is::<MultiBroadcastTo>() && prec.outputs[0].successors.len() == 1 {
            let mut patch = TypedModelPatch::default();
            let mut wires = patch.taps(model, &node.inputs)?;
            wires[ix] = patch.tap_model(model, prec.inputs[0])?;
            let wire = patch.wire_node(&node.name, op.clone(), &wires)?[0];
            patch.shunt_outside(model, node.id.into(), wire)?;
            return Ok(Some(patch));
        }
    }
    Ok(None)
}

/// Rewrite an EinSum whose contraction product is statically 1 as a broadcast Mul.
///
/// Triggers when:
/// - All "k-like" axes (present in both inputs, absent from output) have shape 1 in both inputs, OR
/// - There are no k-like axes at all (Hadamard products like `mn,mn->mn`, outer products like
///   `m,n->mn`, or any pure broadcast pattern).
///
/// In both cases the einsum has no real contraction work — it's a broadcast multiplication
/// dressed up as an einsum. Lowering it as a matmul leaves the GEMM kernel running per-tile
/// setup (clear, panel-load, store) for at most one FMA, so a direct broadcast Mul is much
/// faster on Native (and a net semantic simplification regardless of perf).
///
/// Quantized einsums are left untouched: the existing `dequant` path in `EinSumMatMul::codegen`
/// produces a non-q einsum that this rule then catches naturally on the next declutter pass.
fn unit_k_to_broadcast_mul(
    op: &EinSum,
    model: &TypedModel,
    node: &TypedNode,
) -> TractResult<Option<TypedModelPatch>> {
    if op.q_params.is_some() || node.inputs.len() != 2 {
        return Ok(None);
    }
    let input_facts = model.node_input_facts(node.id)?;
    let input_shapes = op.actual_input_shapes_from_facts(&input_facts)?;
    let k_axes: TVec<&Axis> = op
        .axes
        .iter_all_axes()
        .filter(|a| a.inputs[0].len() == 1 && a.inputs[1].len() == 1 && a.outputs[0].is_empty())
        .collect();
    // Bail if any k-axis is non-trivial — that's a real contraction, leave it to matmul lowering.
    let any_nontrivial_k = k_axes.iter().any(|a| {
        !input_shapes[0][a.inputs[0][0]].is_one() || !input_shapes[1][a.inputs[1][0]].is_one()
    });
    if any_nontrivial_k {
        return Ok(None);
    }
    // Scope: only fire when this einsum's output is consumed by a DeconvSum (i.e. it was
    // emitted by the ConvTranspose lowering pipeline in `Deconv::wire_with_deconv_sum`).
    // That's the original target case (DFN3 / GTCRN depthwise ConvTranspose with 1×N kernel
    // collapsing to K=1 — see PR #2183). Other K=1 einsums (e.g. degenerate Q@K^T inside
    // SDPA when head_dim=1, random-shape proptests with K=1) are intentionally left alone:
    // backend-specific pipelines (Metal SDPA fusion, MetalMul rank-4 broadcast-segment limit,
    // …) pattern-match on the matmul shape and break when we substitute a Mul.
    let has_deconv_sum_consumer = node.outputs.first().map_or(false, |o| {
        o.successors.iter().any(|inlet| model.node(inlet.node).op.name() == "DeconvSum")
    });
    if !has_deconv_sum_consumer {
        return Ok(None);
    }

    let one = TDim::one();
    // Reject "non-trivial single-side disappearing" axes — those need a real reduction.
    for axis in op.axes.iter_all_axes() {
        let in_left =
            axis.inputs[0].first().map(|pos| &input_shapes[0][*pos]).unwrap_or(&one) != &one;
        let in_right =
            axis.inputs[1].first().map(|pos| &input_shapes[1][*pos]).unwrap_or(&one) != &one;
        let in_out = !axis.outputs[0].is_empty();
        if (in_left ^ in_right) && !in_out {
            return Ok(None);
        }
    }

    let c_axes: Vec<char> = op.axes.axes(InOut::Out(0)).map(|a| a.repr).collect();
    if c_axes.is_empty() {
        return Ok(None);
    }

    let k_reprs: TVec<char> = k_axes.iter().map(|a| a.repr).collect();
    let mut patch = TypedModelPatch::new("EinSum unit-K → broadcast Mul");
    let mut wires: TVec<OutletId> = patch.taps(model, &node.inputs)?;
    let name = &node.name;

    for (slot, wire) in wires.iter_mut().enumerate() {
        // Promote inputs to operating_dt so the result type matches EinSum::output_facts
        // (e.g. i8 inputs with i32 operating_dt for an integer matmul that has been dequantized).
        let cur_dt = patch.outlet_fact(*wire)?.datum_type;
        if cur_dt != op.operating_dt {
            *wire = patch.wire_node(
                format!("{name}.cast_in{slot}"),
                crate::ops::cast::cast(op.operating_dt),
                &[*wire],
            )?[0];
        }

        // Drop k axes (sorted descending so positions stay valid).
        let mut k_positions: Vec<usize> = k_axes.iter().map(|a| a.inputs[slot][0]).collect();
        k_positions.sort_by(|a, b| b.cmp(a));
        for (i, pos) in k_positions.into_iter().enumerate() {
            *wire =
                patch.wire_node(format!("{name}.rm_k_in{slot}.{i}"), AxisOp::Rm(pos), &[*wire])?[0];
        }

        let mut current: Vec<char> = op
            .axes
            .axes(InOut::In(slot))
            .map(|a| a.repr)
            .filter(|c| !k_reprs.contains(c))
            .collect();

        // Drop any remaining axes not in output (must be size 1 by precondition above).
        let mut to_drop: Vec<(usize, char)> = current
            .iter()
            .enumerate()
            .filter(|(_, c)| !c_axes.contains(c))
            .map(|(i, c)| (i, *c))
            .collect();
        to_drop.sort_by(|a, b| b.0.cmp(&a.0));
        for (pos, c) in to_drop {
            *wire = patch.wire_node(
                format!("{name}.rm_extra_in{slot}_{c}"),
                AxisOp::Rm(pos),
                &[*wire],
            )?[0];
            current.remove(pos);
        }

        // Insert unit axes for output axes missing from this input.
        for (target_pos, &t) in c_axes.iter().enumerate() {
            if !current.contains(&t) {
                *wire = patch.wire_node(
                    format!("{name}.add_in{slot}_{t}"),
                    AxisOp::Add(target_pos),
                    &[*wire],
                )?[0];
                current.insert(target_pos, t);
            }
        }

        // Permute to match output axis order.
        for (target_pos, &t) in c_axes.iter().enumerate() {
            let cur_pos = current.iter().position(|&c| c == t).unwrap();
            if cur_pos != target_pos {
                *wire = patch.wire_node(
                    format!("{name}.move_in{slot}_{t}"),
                    AxisOp::Move(cur_pos, target_pos),
                    &[*wire],
                )?[0];
                let removed = current.remove(cur_pos);
                current.insert(target_pos, removed);
            }
        }
    }

    let result = patch.wire_node(name, crate::ops::math::mul(), &wires)?;
    patch.shunt_outside(model, node.id.into(), result[0])?;
    Ok(Some(patch))
}
