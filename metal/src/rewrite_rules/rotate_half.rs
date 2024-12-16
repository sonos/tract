use crate::rewrite_rules::{previous_node, previous_nodes};
use crate::rule_ensure;
use tract_core::internal::*;
use tract_core::ops::array::{Slice, TypedConcat};
use tract_core::ops::element_wise::ElementWiseOp;
use tract_core::ops::math::Neg;

#[derive(Clone, Debug, Hash)]
pub struct BasicRotateHalf;

impl Op for BasicRotateHalf {
    fn name(&self) -> Cow<str> {
        "BasicRotateHalf".to_string().into()
    }
    op_as_typed_op!();
}

impl EvalOp for BasicRotateHalf {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let shape: TVec<_> = input.shape().into();
        let mut tensor = Tensor::zero_dt(input.datum_type(), &shape)?;

        let axis = shape.len() - 1;
        ensure!(shape[axis] % 2 == 0, "BasicRotateHalf possible only if the most inner dimension of the shape {:?} is divible by 2", shape);
        let half = shape[axis] / 2;
        unsafe { tensor.assign_slice_unchecked(0..half, &input, half.., axis) };
        Neg {}.eval_in_place(&mut tensor, None)?;
        unsafe { tensor.assign_slice_unchecked(half.., &input, 0..half, axis) };
        Ok(tvec![tensor.into()])
    }
}

impl TypedOp for BasicRotateHalf {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let dt = inputs[0].datum_type;
        let fact = dt.fact(inputs[0].shape.clone());
        Ok(tvec!(fact))
    }

    as_op!();
}

/// Search pattern:
/// Y = Concat(Neg(Slice(X, X.shape[-1]/2.., -1)), Slice(X, ..X.shape[-1]/2, -1))
pub fn as_rotate_half_rule(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    op: &TypedConcat,
) -> TractResult<Option<TypedModelPatch>> {
    let out_fact = model.node_output_facts(node.id)?[0];
    let dt = out_fact.datum_type;
    rule_ensure!(dt.is_float() || dt.is_integer());
    rule_ensure!(op.axis == out_fact.rank() - 1);

    let in_concat = previous_nodes(model, node);
    rule_ensure!(in_concat.len() == 2);

    let neg_half = in_concat[0];
    let Some(neg_half_op) = neg_half.op_as::<ElementWiseOp>() else { return Ok(None) };
    rule_ensure!(neg_half_op.0.is::<Neg>());

    let Some(neg_half_slice) = previous_node(model, neg_half) else { return Ok(None) };
    let Some(neg_half_slice_op) = neg_half_slice.op_as::<Slice>() else { return Ok(None) };

    rule_ensure!(neg_half_slice_op.axis == op.axis);

    let pos_half = in_concat[1];
    let Some(pos_half_op) = pos_half.op_as::<Slice>() else { return Ok(None) };

    rule_ensure!(pos_half_op.axis == op.axis);
    rule_ensure!(pos_half_op.end == neg_half_slice_op.start);
    rule_ensure!(neg_half_slice_op.end == out_fact.shape[op.axis].clone());

    // Ensure it is a half rotation
    let Some(pos_half_slice_end) = pos_half_op.end.as_i64() else { return Ok(None) };
    let Some(concatenated_last_dim) = out_fact.shape[op.axis].as_i64() else { return Ok(None) };
    rule_ensure!(pos_half_slice_end * 2 == concatenated_last_dim);

    let in_fact = model.node_input_facts(neg_half_slice.id)?[0];

    let mut patch = TypedModelPatch::default();
    let mut inputs = patch.taps(model, &neg_half_slice.inputs)?;

    if pos_half_op.start != 0.into() || neg_half_slice_op.end != in_fact.shape[op.axis] {
        inputs = patch.wire_node(
            format!("{node_name}.rotate_half.slice"),
            Slice {
                start: pos_half_op.start.clone(),
                end: neg_half_slice_op.end.clone(),
                axis: op.axis,
            },
            &inputs,
        )?;
    }

    let out = patch.wire_node(format!("{node_name}.rotate_half"), BasicRotateHalf, &inputs)?;
    patch.shunt_outside(model, node.id.into(), out[0])?;

    Ok(Some(patch))
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::AsPrimitive;
    use num_traits::Zero;

    fn run_test_case<F: Datum + Zero + Copy>(a_shape: &[usize]) -> TractResult<()>
    where
        usize: AsPrimitive<F>,
    {
        let a_len = a_shape.iter().product::<usize>();
        let input = Tensor::from_shape(a_shape, &(0..a_len).map(|f| f.as_()).collect::<Vec<F>>())?;
        let rotated = BasicRotateHalf.eval(tvec![input.clone().into()])?;
        let mut back = args_1!(BasicRotateHalf.eval(rotated)?).into_tensor();
        Neg {}.eval_in_place(&mut back, None)?;
        back.close_enough(&input, Approximation::Close)?;
        Ok(())
    }

    #[test]
    fn test_rotate_half() -> TractResult<()> {
        run_test_case::<f32>(&[2, 2])?;
        run_test_case::<f32>(&[512, 512])?;
        run_test_case::<f32>(&[10, 512, 1024])?;

        Ok(())
    }
}
