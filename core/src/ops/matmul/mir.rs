use crate::internal::*;
use crate::ops::array::Slice;
use crate::ops::matmul::*;

/// The binary op. It will declutter to MatMulUnary if either A or B is constant.
///
/// TODO: implemnent TypedOp fully to play nice with optimizer.
/// TODO: codegen fails if A and B are variable inputs.
#[derive(Debug, Clone, Default, Hash)]
pub struct MatMul {
    pub axes: MatMulAxes,
}

impl_dyn_hash!(MatMul);

impl Op for MatMul {
    fn name(&self) -> Cow<str> {
        "MatMul".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("{:?}", self.axes)])
    }

    op_as_typed_op!();
}

impl EvalOp for MatMul {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        if inputs[0].rank() != inputs[1].rank() {
            bail!("Rank mismatch {:?} vs {:?}", inputs[0], inputs[1]);
        }
        Ok(tvec!(eval(&inputs[0], &inputs[1], self.axes)?.into()))
    }
}

impl TypedOp for MatMul {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        if inputs[0].rank() != inputs[1].rank() {
            bail!(
                "Inconsistent matmul between {:?} and {:?} (rank mismatch)",
                inputs[0],
                inputs[1]
            );
        }
        let (_m, _k, _n, c_shape) = compute_shape(&inputs[0].shape, &inputs[1].shape, self.axes)?;
        Ok(tvec!(output_type(inputs[0].datum_type).fact(c_shape)))
    }

    fn invariants(&self, inputs: &[&TypedFact], outputs: &[&TypedFact]) -> TractResult<Invariants> {
        mir_invariants(inputs[0], inputs[1], outputs[0], self.axes)
    }

    fn change_axes(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        if let Some((axes, wire_changes)) = mir_change_axes(model, node, io, change, &self.axes)? {
            let op = Self { axes };
            Ok(Some(AxisChangeConsequence { substitute_op: Some(Box::new(op)), wire_changes }))
        } else {
            Ok(None)
        }
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        if let Some(boundaries) =
            crate::ops::matmul::mir_unary::should_slice_output(model, node, self.axes.c_m)?
        {
            let mut patch = TypedModelPatch::new("split over m-concatenated output");
            let a = patch.tap_model(model, node.inputs[0])?;
            let b = patch.tap_model(model, node.inputs[1])?;
            let mut start = 0;
            let mut splits = tvec!();
            for end in &boundaries {
                let wire = patch.wire_node(
                    format!("{}.split-a-over-m.{}..{}.slice", &node.name, start, end),
                    Slice { axis: self.axes.a_m, start: start.to_dim(), end: end.to_dim() },
                    &[a],
                )?;
                let wire = patch.wire_node(
                    format!("{}.split-a-over-m.{}..{}.mm", &node.name, start, end),
                    self.clone(),
                    &[wire[0], b],
                )?;
                splits.push(wire[0]);
                start = *end;
            }
            crate::ops::matmul::mir_unary::rewire_sliced_outputs(
                model,
                node,
                self.axes.c_m,
                &mut patch,
                &boundaries,
                &splits,
            )?;
            Ok(Some(patch))
        } else if let Some(boundaries) =
            crate::ops::matmul::mir_unary::should_slice_output(model, node, self.axes.c_n)?
        {
            let mut patch = TypedModelPatch::new("split over n-concatenated output");
            let a = patch.tap_model(model, node.inputs[0])?;
            let b = patch.tap_model(model, node.inputs[1])?;
            let mut start = 0;
            let mut splits = tvec!();
            for end in &boundaries {
                let wire = patch.wire_node(
                    format!("{}.split-b-over-n.{}..{}.slice", &node.name, start, end),
                    Slice { axis: self.axes.b_n, start: start.to_dim(), end: end.to_dim() },
                    &[b],
                )?;
                let wire = patch.wire_node(
                    format!("{}.split-b-over-n.{}..{}.mm", &node.name, start, end),
                    self.clone(),
                    &[a, wire[0]],
                )?;
                splits.push(wire[0]);
                start = *end;
            }
            crate::ops::matmul::mir_unary::rewire_sliced_outputs(
                model,
                node,
                self.axes.c_n,
                &mut patch,
                &boundaries,
                &splits,
            )?;
            Ok(Some(patch))
        } else {
            Ok(None)
        }
    }

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let a_fact = model.outlet_fact(node.inputs[0])?;
        let b_fact = model.outlet_fact(node.inputs[1])?;
        let konst_ix = if a_fact.konst.is_some() {
            0
        } else if b_fact.konst.is_some() {
            1
        } else {
            return Ok(None);
        };

        let var_ix = 1 - konst_ix;
        let flip = konst_ix == 1;

        let konst_fact = [a_fact, b_fact][konst_ix];

        let axes = if flip {
            MatMulAxes {
                a_m: self.axes.b_n,
                a_k: self.axes.b_k,
                b_n: self.axes.a_m,
                b_k: self.axes.a_k,
                c_m: self.axes.c_n,
                c_n: self.axes.c_m,
            }
        } else {
            self.axes
        };

        let konst = konst_fact.konst.as_ref().unwrap();
        crate::ops::matmul::mir_unary::new_mat_mul_unary_finite(model, node, konst, var_ix, &axes)
            .map(Some)
    }

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        super::cost(
            &inputs[0].shape.to_tvec(),
            &inputs[1].shape.to_tvec(),
            inputs[0].datum_type,
            self.axes,
        )
    }

    as_op!();
}

fn mir_invariants(
    _a: &TypedFact,
    _b: &TypedFact,
    c: &TypedFact,
    axes: MatMulAxes,
) -> TractResult<Invariants> {
    Ok((0..c.rank())
        .map(|c_axis| {
            if c_axis == axes.c_m {
                AxisInfo {
                    inputs: tvec!(Some(axes.a_m), None),
                    outputs: tvec!(Some(c_axis)),
                    disposable: false,
                    period: 1,
                }
            } else if c_axis == axes.c_n {
                AxisInfo {
                    inputs: tvec!(None, Some(axes.b_n)),
                    outputs: tvec!(Some(c_axis)),
                    disposable: false,
                    period: 1,
                }
            } else {
                let tracking = axes.follow_axis_from_c(c_axis);
                AxisInfo {
                    inputs: tvec!(Some(tracking.0), Some(tracking.1)),
                    outputs: tvec!(Some(c_axis)),
                    disposable: true,
                    period: 1,
                }
            }
        })
        .collect())
}

#[allow(clippy::type_complexity)]
pub(super) fn mir_change_axes(
    model: &TypedModel,
    node: &TypedNode,
    io: InOut,
    change: &AxisOp,
    old_axes: &MatMulAxes,
) -> TractResult<Option<(MatMulAxes, TVec<(InOut, AxisOp)>)>> {
    let rank = model.outlet_fact(node.inputs[0])?.rank();
    let result = if io == InOut::In(0) {
        old_axes.change_axis_from_a(change, rank)
    } else if io == InOut::In(1) {
        old_axes.change_axis_from_b(change, rank)
    } else if io == InOut::Out(0) {
        old_axes.change_axis_from_c(change, rank)
    } else {
        unreachable!();
    };
    if let Ok((axes, change_a, change_b, change_c)) = result {
        let mut wires = tvec!();
        if let Some(change_a) = change_a {
            wires.push((InOut::In(0), change_a));
        }
        if let Some(change_b) = change_b {
            wires.push((InOut::In(1), change_b));
        }
        if let Some(change_c) = change_c {
            wires.push((InOut::Out(0), change_c));
        }
        Ok(Some((axes, wires)))
    } else {
        Ok(None) // is it right ? or return error ?
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn bin() {
        //           0
        //           1
        //           2
        //
        // 0 1 2     5
        // 3 4 5    14
        let a = tensor2(&[[0f32, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let b = tensor2(&[[0f32], [1.0], [2.0]]);
        let c = tensor2(&[[5f32], [14.0]]);
        let op = MatMul::default();
        let c_found = op.eval(tvec!(a.into(), b.into())).unwrap().pop().unwrap();
        c.close_enough(&c_found, true).unwrap();
    }

    #[test]
    fn bin_transpose() {
        let a = tensor2(&[[0f32, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let b = tensor2(&[[0f32], [1.0], [2.0]]);
        let c = tensor2(&[[5f32], [14.0]]);
        let op = MatMul { axes: MatMulAxes::default().transposing(true, true, true) };
        let c_found = op.eval(tvec!(b.into(), a.into())).unwrap().pop().unwrap();

        c.close_enough(&c_found, true).unwrap();
    }

    #[test]
    fn batch_input() -> TractResult<()> {
        crate::setup_test_logger();
        let (batch, len, ci, co) = (2, 3, 4, 5);
        let mut model = TypedModel::default();
        let input_shape = tvec!(batch, len, ci);
        let mut wire = tvec!(model.add_source("s", f32::fact(&*input_shape))?);
        let mut a = Tensor::zero::<f32>(&[1, ci, co])?;
        a.as_slice_mut::<f32>().unwrap()[0] = 1.0;
        let a = a.into_arc_tensor();
        wire = model.wire_node(
            "m",
            MatMulUnary { a, axes: MatMulAxes::default_for_rank(3).transposing(true, true, true) },
            &wire,
        )?;
        let mut b = Tensor::zero::<f32>(&[1, 1, co])?;
        b.as_slice_mut::<f32>().unwrap()[0] = 1.0;
        let b = b.into_arc_tensor();
        wire = model.wire_node("a", crate::ops::math::add::unary(b), &wire)?;
        model.set_output_outlets(&wire)?;
        let input = Tensor::zero::<f32>(&input_shape)?.into_tvalue();
        trace!("running mir");
        model.clone().into_runnable()?.run(tvec!(input.clone()))?;
        trace!("running optimized");
        model.into_decluttered()?.into_optimized()?.into_runnable()?.run(tvec!(input))?;
        Ok(())
    }
}
