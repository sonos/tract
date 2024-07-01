use std::fmt::Debug;

use crate::internal::*;
use crate::ops::array::Slice;
use crate::tract_data::itertools::Itertools;

mod eval;

#[cfg(feature="blas")]
pub mod as_blas;
use super::array::TypedConcat;
use super::math::add;
mod as_matmul;
pub mod codegen;

#[cfg(test)]
mod proptest;

pub use as_matmul::{rewrite_einsums_as_matmul, BasicMatMul};

#[derive(Clone, Hash)]
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

    #[allow(clippy::comparison_chain)]
    fn declutter_after_concat(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if self.q_params.is_some() {
            // FIXME
            return Ok(None);
        }
        'outer: for (slot, input) in node.inputs.iter().enumerate() {
            let precursor = model.node(input.node);
            if let Some(concat) = precursor.op_as::<TypedConcat>() {
                let offsets = concat.offsets(&model.node_input_facts(precursor.id)?)?;
                let axis_info = self.axes.axis((InOut::In(slot), concat.axis))?;
                // only split if axis is a summing axis
                if axis_info.outputs[0].len() > 0 {
                    continue;
                }
                let mut patch = TypedModelPatch::new(format!(
                    "Split Einsum for concat on axis {}",
                    axis_info.repr
                ));
                // inputs[einsum_input_slot][concated_slice]. concated_slice = 0 for broadcast
                let mut inputs: TVec<TVec<OutletId>> = tvec!();
                for (slot, input) in node.inputs.iter().enumerate() {
                    let tap = patch.tap_model(model, *input)?;
                    if axis_info.inputs[slot].len() > 1 {
                        continue 'outer;
                    } else if axis_info.inputs[slot].len() == 1 {
                        let mut slices = tvec!();
                        for (start, end) in offsets.iter().cloned().tuple_windows() {
                            let wire = patch.wire_node(
                                format!(
                                    "{}.concat-einsum-slice-{}.{}.{}..{}",
                                    node.name, axis_info.repr, slot, start, end
                                ),
                                Slice { axis: axis_info.inputs[slot][0], start, end },
                                &[tap],
                            )?;
                            slices.push(wire[0]);
                        }
                        inputs.push(slices);
                    } else {
                        inputs.push(tvec!(tap)); // broadcast
                    };
                }
                let mut einsums = tvec!();
                for (ix, (start, end)) in offsets.iter().tuple_windows().enumerate() {
                    let mut einsum_inputs = tvec!();
                    for input_ix in 0..node.inputs.len() {
                        einsum_inputs
                            .push(inputs[input_ix].get(ix).cloned().unwrap_or(inputs[input_ix][0]));
                    }
                    let einsum = patch.wire_node(
                        format!(
                            "{}.concat-einsum-{}.{}..{}",
                            node.name, axis_info.repr, start, end
                        ),
                        self.clone(),
                        &einsum_inputs,
                    )?[0];
                    einsums.push(einsum);
                }
                let wire = if let Some(axis) = axis_info.outputs[0].first().cloned() {
                    patch.wire_node(
                        format!("{}.concat-einsum-{}.concat", node.name, axis_info.repr),
                        TypedConcat { axis },
                        &einsums,
                    )?[0]
                } else {
                    let mut wire = einsums[0];
                    for ix in 1..einsums.len() {
                        wire = patch.wire_node(
                            format!("{}.concat-einsum-{}.add-{}", node.name, axis_info.repr, ix),
                            add(),
                            &[wire, einsums[ix]],
                        )?[0]
                    }
                    wire
                };
                patch.shunt_outside(model, node.id.into(), wire)?;
                return Ok(Some(patch));
            }
        }
        Ok(None)
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
        ensure!(inputs.len() == self.axes.input_count());
        ensure!(inputs
            .iter()
            .enumerate()
            .all(|(ix, fact)| fact.rank() == self.axes.rank(InOut::In(ix))));
        let shapes: TVec<&[TDim]> = inputs.iter().map(|t| &*t.shape).collect();
        if let Some(qp) = self.q_params {
            ensure!(inputs.len() == 9);
            Ok(tvec!(qp.fact(eval::output_shape(&self.axes, &shapes[0..2]))))
        } else {
            Ok(tvec!(TypedFact::dt_shape(
                self.operating_dt,
                eval::output_shape(&self.axes, &shapes)
            )))
        }
    }

    fn axes_mapping(
        &self,
        _inputs: &[&TypedFact],
        _outputs: &[&TypedFact],
    ) -> TractResult<AxesMapping> {
        Ok(self.axes.clone())
    }

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        let shapes: TVec<&[TDim]> = inputs.iter().map(|t| &*t.shape).collect();
        let oshape = eval::output_shape(&self.axes, &shapes);
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
        prefix: &str,
        inputs: &[OutletId],
        _output_axis: usize,
        _start: usize,
        _end: usize,
    ) -> TractResult<Option<TVec<OutletId>>> {
        patch.wire_node(prefix, self.clone(), inputs).map(Some)
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
            _ => return Ok(None),
        };
        *interface = axes.into_iter().collect();
        let axes = AxesMapping::from_strs(&inputs, &outputs)?;
        Ok(Some(AxisChangeConsequence {
            substitute_op: Some(Box::new(EinSum { axes, ..self.clone() })),
            wire_changes: tvec!((io, change.clone())),
        }))
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        self.declutter_after_concat(model, node)
    }

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        codegen::codegen(self, model, node)
    }

    as_op!();
}
