use std::borrow::Borrow;
use std::fmt::Debug;

use crate::internal::*;
use crate::ops::array::MultiBroadcastTo;
use crate::tract_data::itertools::Itertools;

mod eval;

#[cfg(feature = "blas")]
pub mod as_blas;
pub mod einsum_matmul;
pub mod kernel_selection;
pub mod prefix_matmul;

#[cfg(test)]
mod proptest;

use num_traits::One;
use tract_linalg::block_quant::{BlockQuantFact, PackedBlockQuantFact};
use tract_linalg::mmm::PackedOpaqueFact;

pub fn block_quant_aware_input_shape(fact: &TypedFact) -> TractResult<Cow<[TDim]>> {
    if !fact.datum_type.is_opaque() {
        return Ok(Cow::Borrowed(&*fact.shape));
    }
    let Some(opaque_fact) = fact.opaque_fact() else {
        bail!("Datum fact is opaque, but no opaque fact was found.")
    };
    if let Some(bqf) = opaque_fact.downcast_ref::<BlockQuantFact>() {
        Ok(Cow::Owned(
            fact.shape.iter().cloned().chain(bqf.shape().iter().map(|d| d.to_dim())).collect_vec(),
        ))
    } else if let Some(pof) = opaque_fact.downcast_ref::<PackedBlockQuantFact>() {
        Ok(Cow::Owned(
            fact.shape.iter().cloned().chain(pof.shape.iter().map(|i| i.to_dim())).collect_vec(),
        ))
    } else if let Some(pof) = opaque_fact.downcast_ref::<PackedOpaqueFact>() {
        Ok(Cow::Owned(
            fact.shape.iter().cloned().chain([pof.mn.clone(), pof.k.to_dim()]).collect_vec(),
        ))
    } else {
        bail!("Unsupported opaque fact {opaque_fact:?}")
    }
}

#[derive(Clone, Hash, PartialEq)]
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
        ensure!(shapes
            .iter()
            .enumerate()
            .all(|(ix, fact)| fact.len() == self.axes.rank(InOut::In(ix))));
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
            if new_axis.inputs[ix].len() > 1 {
                return Ok(None); // FIXME maybe
            } else if new_axis.inputs[ix].is_empty() {
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
    fn name(&self) -> Cow<str> {
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
            assert!(shapes
                .iter()
                .enumerate()
                .flat_map(|(slot, shape)| axis.inputs[slot].iter().map(|a| &shape[*a]))
                .try_fold(TDim::one(), |a, b| TDim::broadcast(a, b.clone()))
                .is_ok());
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

    fn axes_mapping(
        &self,
        inputs: &[&TypedFact],
        _outputs: &[&TypedFact],
    ) -> TractResult<AxesMapping> {
        let mut axes = self.axes.clone();
        for (slot, i) in inputs.iter().enumerate() {
            if i.datum_type.is_opaque()
                && (i.opaque_fact().is_some_and(|of| {
                    of.is::<BlockQuantFact>()
                        || of.is::<PackedOpaqueFact>()
                        || of.is::<PackedBlockQuantFact>()
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
            .any(|(slot, fact)| axis.inputs[slot].len() > 0 && fact.datum_type.is_opaque())
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
        Ok(None)
    }

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if (self.q_params.is_none() && node.inputs.len() != 2)
            || (self.q_params.is_some() && node.inputs.len() != 9)
        {
            return Ok(None);
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
        let Some(AxisOp::Reshape(at, ref from, ref to)) = prec.op_as() else { continue };
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
        let folded_axis = op.axes.axis((InOut::In(slot), *at))?;
        if folded_axis.outputs[0].len() > 1 {
            return Ok(None);
        };
        let mut patch = TypedModelPatch::default();
        let mut taps = patch.taps(model, &node.inputs)?;
        for (input, tap) in taps.iter_mut().enumerate() {
            if folded_axis.inputs[input].len() == 0 {
                continue;
            };
            if folded_axis.inputs[input].len() > 1 {
                return Ok(None);
            };
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
