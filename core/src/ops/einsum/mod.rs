use std::fmt::Debug;

use crate::internal::*;
use crate::ops::array::Slice;
use crate::tract_data::itertools::Itertools;

mod eval;
pub use crate::axes::*;

use super::array::TypedConcat;
use super::math::add;
// mod to_matmul;

#[derive(Clone, Hash, new)]
pub struct EinSum {
    pub expr: AxesMapping,
    pub operating_dt: DatumType,
}

impl EinSum {
    #[allow(unused_variables)]
    pub(crate) fn propagate_axis(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        io: InOut,
        axis: usize,
    ) -> TractResult<Option<TypedModelPatch>> {
        let mut new_axis = match io {
            InOut::In(slot) => self.expr.input_axis(slot, axis).unwrap().clone(),
            InOut::Out(_) => self.expr.output_axis(0, axis).unwrap().clone(),
        };
        let repr = new_axis.repr;
        let mut patch = TypedModelPatch::new(format!("Propagate axis {}", new_axis.repr));
        let mut taps = tvec!();
        for (ix, input) in node.inputs.iter().enumerate() {
            let mut tap = patch.tap_model(model, *input)?;
            if new_axis.inputs[ix].len() > 1 {
                return Ok(None); // FIXME maybe
            } else if new_axis.inputs[ix].is_empty() {
                let insert_at = self.expr.input_rank(ix);
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
            let insert_at = self.expr.output_rank(0);
            new_axis.outputs[0].push(insert_at);
            Some(insert_at)
        } else {
            None
        };
        let new_expr: AxesMapping = self
            .expr
            .iter_all_axes()
            .map(|it| if it.repr == new_axis.repr { new_axis.clone() } else { it.clone() })
            .collect();
        let mut wire =
            patch.wire_node(&node.name, Self { expr: new_expr, ..self.clone() }, &taps)?;
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

    fn declutter_after_concat(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        'outer: for (slot, input) in node.inputs.iter().enumerate() {
            let precursor = model.node(input.node);
            if let Some(concat) = precursor.op_as::<TypedConcat>() {
                let offsets = concat.offsets(&model.node_input_facts(precursor.id)?)?;
                let axis_info = self.expr.input_axis(slot, concat.axis).context("Axis unmapped")?;
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
                let wire = if let Some(axis) = axis_info.outputs[0].get(0).cloned() {
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
        return Ok(None);
    }
}

impl Debug for EinSum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "EinSum {} ({:?})", self.expr, self.operating_dt)
    }
}

impl Op for EinSum {
    fn name(&self) -> Cow<str> {
        "EinSum".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("{}", self.expr)])
    }

    op_as_typed_op!();
}

impl EvalOp for EinSum {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        dispatch_numbers!(eval::eval_t(self.operating_dt)(&self.expr, inputs))
    }
}

impl TypedOp for EinSum {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.len() == self.expr.input_count());
        ensure!(inputs
            .iter()
            .enumerate()
            .all(|(ix, fact)| fact.rank() == self.expr.input_rank(ix)));
        let shapes: TVec<&[TDim]> = inputs.iter().map(|t| &*t.shape).collect();
        Ok(tvec!(TypedFact::dt_shape(self.operating_dt, eval::output_shape(&self.expr, &shapes))))
    }

    fn invariants(&self, inputs: &[&TypedFact], outputs: &[&TypedFact]) -> TractResult<Invariants> {
        let inv = self
            .expr
            .iter_all_axes()
            .filter_map(|axis| {
                // if axis is used twice, don't even dare do anything
                if axis.inputs.iter().any(|input| input.len() > 1) {
                    None
                } else {
                    let i = (0..inputs.len()).map(|i| axis.inputs[i].get(0).cloned()).collect();
                    let o = (0..outputs.len()).map(|i| axis.outputs[i].get(0).cloned()).collect();
                    Some(AxisInfo { inputs: i, outputs: o})
                }
            })
            .collect();
        Ok(inv)
    }

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        let shapes: TVec<&[TDim]> = inputs.iter().map(|t| &*t.shape).collect();
        let oshape = eval::output_shape(&self.expr, &shapes);
        let ks = self
            .expr
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
                    .max()
                    .unwrap()
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
        let (mut inputs, mut outputs) = self.expr.to_strs();
        let interface: &mut String = match io {
            InOut::In(i) => &mut inputs[i],
            InOut::Out(o) => &mut outputs[o],
        };
        let mut axes: Vec<char> = interface.chars().collect();
        match change {
            AxisOp::Rm(rm) => {
                axes.remove(*rm);
            }
            AxisOp::Add(add) => axes.insert(*add, self.expr.available_label()),
            AxisOp::Move(from, to) => {
                let c = axes.remove(*from);
                axes.insert(*to, c);
            }
            _ => return Ok(None),
        };
        *interface = axes.into_iter().collect();
        let expr = AxesMapping::from_strs(&inputs, &outputs);
        return Ok(Some(AxisChangeConsequence {
            substitute_op: Some(Box::new(EinSum::new(expr, self.operating_dt))),
            wire_changes: tvec!((io, change.clone())),
        }));
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        self.declutter_after_concat(model, node)
    }

    as_op!();
}
