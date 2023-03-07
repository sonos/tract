use crate::internal::*;
use crate::ops::einsum::EinSum;

use super::{compute_shape, eval, output_type, MatMulAxes};

/// The binary op. It will declutter to MatMulUnary if either A or B is constant.
#[derive(Debug, Clone, Default, Hash)]
pub struct MatMul {
    pub axes: MatMulAxes,
}

impl Op for MatMul {
    fn name(&self) -> Cow<str> {
        "MatMul".into()
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
        anyhow::ensure!(inputs.len() == 2, "MatMul must have two inputs");
        anyhow::ensure!(
            inputs[0].rank() == inputs[1].rank(),
            "MatMul inputs must have the same rank"
        );
        let (_m, _k, _n, c_shape) = compute_shape(&inputs[0].shape, &inputs[1].shape, self.axes)?;
        Ok(tvec!(output_type(inputs[0].datum_type).fact(c_shape)))
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let a_fact = model.outlet_fact(node.inputs[0])?;
        let b_fact = model.outlet_fact(node.inputs[1])?;
        assert!(a_fact.rank() == b_fact.rank());
        let axes = self.axes.to_axis_mapping(a_fact.rank())?;
        TypedModelPatch::replace_single_op(
            model,
            node,
            &node.inputs,
            EinSum::new(axes, output_type(a_fact.datum_type), None),
        )
        .map(Some)
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

#[allow(clippy::type_repetition_in_bounds, clippy::type_complexity)]
pub(super) fn mir_change_axes(
    model: &TypedModel,
    node: &TypedNode,
    io: InOut,
    change: &AxisOp,
    old_axes: &MatMulAxes,
) -> TractResult<Option<(MatMulAxes, TVec<(InOut, AxisOp)>)>> {
    let a_fact = model.outlet_fact(node.inputs[0])?;
    let result = if io == InOut::In(1) {
        old_axes.change_axis_from_b(change, a_fact.rank())
    } else if io == InOut::Out(0) {
        old_axes.change_axis_from_c(change, a_fact.rank())
    } else {
        return Ok(None);
    };
    if let Ok((axes, change_a, change_b, change_c)) = result {
        let mut changes = tvec!();
        if let Some(change) = change_a {
            changes.push((InOut::In(0), change))
        }
        if let Some(change) = change_b {
            changes.push((InOut::In(1), change))
        }
        if let Some(change) = change_c {
            changes.push((InOut::Out(0), change))
        }
        Ok(Some((axes, changes)))
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
        let a = model.add_const("a", a)?;
        wire = model.wire_node(
            "m",
            MatMul { axes: MatMulAxes::default_for_rank(3).transposing(true, true, true) },
            &[a, wire[0]],
        )?;
        let mut b = Tensor::zero::<f32>(&[1, 1, co])?;
        b.as_slice_mut::<f32>().unwrap()[0] = 1.0;
        let b = b.into_arc_tensor();
        let b = model.add_const("b", b)?;
        wire = model.wire_node("c", crate::ops::math::add(), &[wire[0], b])?;
        model.set_output_outlets(&wire)?;
        let input = Tensor::zero::<f32>(&input_shape)?.into_tvalue();
        trace!("running mir");
        model.clone().into_runnable()?.run(tvec!(input.clone()))?;
        trace!("running optimized");
        model.into_decluttered()?.into_optimized()?.into_runnable()?.run(tvec!(input))?;
        Ok(())
    }
}
