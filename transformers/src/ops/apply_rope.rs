use tract_nnef::internal::*;
use tract_nnef::tract_core::ops::array::{Slice, TypedConcat};
use tract_nnef::tract_core::ops::binary::BinMiniOp;
use tract_nnef::tract_core::ops::binary::TypedBinOp;
use tract_nnef::tract_core::ops::element_wise::ElementWiseOp;
use tract_nnef::tract_core::ops::math::{Add, Mul, Neg};

use super::{previous_node, previous_nodes, single_prev_node_as};
use crate::rule_ensure;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(ser_apply_rope);
    registry.register_primitive(
        "tract_transformers_apply_rope",
        &[
            TypeName::Scalar.tensor().named("input"),
            TypeName::Scalar.tensor().named("cos"),
            TypeName::Scalar.tensor().named("sin"),
        ],
        &[("output", TypeName::Scalar.tensor())],
        de_apply_rope,
    );
}

fn de_apply_rope(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<Value> {
    let input = invocation.named_arg_as(builder, "input")?;
    let cos = invocation.named_arg_as(builder, "cos")?;
    let sin = invocation.named_arg_as(builder, "sin")?;
    builder.wire(ApplyRope, &[input, cos, sin])
}

fn ser_apply_rope(
    ast: &mut IntoAst,
    node: &TypedNode,
    _op: &ApplyRope,
) -> TractResult<Option<Arc<RValue>>> {
    let input = ast.mapping[&node.inputs[0]].clone();
    let cos: Arc<RValue> = ast.mapping[&node.inputs[1]].clone();
    let sin: Arc<RValue> = ast.mapping[&node.inputs[2]].clone();
    Ok(Some(invocation("tract_transformers_apply_rope", &[input, cos, sin], &[])))
}

#[derive(Clone, Debug, Hash)]
pub struct RotateHalf;

impl Op for RotateHalf {
    fn name(&self) -> Cow<str> {
        "RotateHalf".to_string().into()
    }
    op_as_typed_op!();
}

impl EvalOp for RotateHalf {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let shape: TVec<_> = input.shape().into();
        let mut tensor = Tensor::zero_dt(input.datum_type(), &shape)?;

        let axis = shape.len() - 1;
        ensure!(shape[axis] % 2 == 0, "RotateHalf possible only if the most inner dimension of the shape {:?} is divible by 2", shape);
        let half = shape[axis] / 2;
        unsafe { tensor.assign_slice_unchecked(0..half, &input, half.., axis) };
        Neg {}.eval_in_place(&mut tensor, None)?;
        unsafe { tensor.assign_slice_unchecked(half.., &input, 0..half, axis) };
        Ok(tvec![tensor.into()])
    }
}

impl TypedOp for RotateHalf {
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

    let out = patch.wire_node(format!("{node_name}.rotate_half"), RotateHalf, &inputs)?;
    patch.shunt_outside(model, node.id.into(), out[0])?;

    Ok(Some(patch))
}

#[derive(Clone, Debug, Hash)]
pub struct ApplyRope;

impl ApplyRope {
    pub fn is_supported_dt(dt: DatumType) -> bool {
        matches!(dt, DatumType::F32 | DatumType::F16)
    }
}

impl Op for ApplyRope {
    fn name(&self) -> Cow<str> {
        "ApplyRope".to_string().into()
    }
    op_as_typed_op!();
}

impl EvalOp for ApplyRope {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let (input, cos, sin) = args_3!(inputs);
        let rotated_input = args_1!(RotateHalf.eval(tvec![input.clone()])?);
        let mul_with_cos = Mul.eval(input.clone(), cos, input.datum_type())?;
        let mul_with_sin = Mul.eval(rotated_input, sin, input.datum_type())?;
        let output = Add.eval(mul_with_cos.into(), mul_with_sin.into(), input.datum_type())?;
        Ok(tvec![output.into()])
    }
}

impl TypedOp for ApplyRope {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let dt = inputs[0].datum_type;
        let fact = dt.fact(inputs[0].shape.clone());
        Ok(tvec!(fact))
    }

    as_op!();
}

/// Search pattern:
/// Y = X * Cos + RotateHalf(X) * Sin
pub fn as_apply_rope_rule(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    op: &TypedBinOp,
) -> TractResult<Option<TypedModelPatch>> {
    rule_ensure!(op.0.is::<Add>());

    let in_add = previous_nodes(model, node);
    rule_ensure!(in_add.len() == 2);

    let cos_mul = in_add[0];
    let Some(cos_mul_op) = cos_mul.op_as::<TypedBinOp>() else { return Ok(None) };
    rule_ensure!(cos_mul_op.0.is::<Mul>());

    let sin_mul = in_add[1];
    let Some(sin_mul_op) = sin_mul.op_as::<TypedBinOp>() else { return Ok(None) };
    rule_ensure!(sin_mul_op.0.is::<Mul>());

    let Some((rotate_half_in_idx, rotate_half)) =
        single_prev_node_as::<RotateHalf>(model, sin_mul)
    else {
        return Ok(None);
    };

    // If cos and rotate half don't share the same input, we check if they don't
    // input node that are the same.
    let (apply_rope_in, cos) = if !cos_mul.inputs.contains(&rotate_half.inputs[0]) {
        let Some(rotate_half_prev) = previous_node(model, rotate_half) else { return Ok(None) };
        let Some((cos_common_input_idx, _)) = previous_nodes(model, cos_mul)
            .iter()
            .enumerate()
            .find(|(_, n)| n.same_as(rotate_half_prev))
        else {
            return Ok(None);
        };
        (rotate_half.inputs[0], cos_mul.inputs[1 - cos_common_input_idx])
    } else {
        let apply_rope_in = rotate_half.inputs[0];
        let cos =
            if cos_mul.inputs[0] == apply_rope_in { cos_mul.inputs[1] } else { cos_mul.inputs[0] };
        (apply_rope_in, cos)
    };

    let sin = sin_mul.inputs[1 - rotate_half_in_idx];

    rule_ensure!(ApplyRope::is_supported_dt(model.outlet_fact(apply_rope_in)?.datum_type));
    rule_ensure!(ApplyRope::is_supported_dt(model.outlet_fact(cos)?.datum_type));
    rule_ensure!(ApplyRope::is_supported_dt(model.outlet_fact(sin)?.datum_type));

    let mut patch = TypedModelPatch::default();
    let input = patch.tap_model(model, apply_rope_in)?;
    let cos = patch.tap_model(model, cos)?;
    let sin = patch.tap_model(model, sin)?;
    let out =
        patch.wire_node(format!("{node_name}.apply_rope"), ApplyRope, &[input, cos, sin])?;
    patch.shunt_outside(model, node.id.into(), out[0])?;
    Ok(Some(patch))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tract_nnef::tract_core::ops::math::Neg;
    use tract_num_traits::AsPrimitive;
    use tract_num_traits::Zero;

    fn run_test_case<F: Datum + Zero + Copy>(a_shape: &[usize]) -> TractResult<()>
    where
        usize: AsPrimitive<F>,
    {
        let a_len = a_shape.iter().product::<usize>();
        let input = Tensor::from_shape(a_shape, &(0..a_len).map(|f| f.as_()).collect::<Vec<F>>())?;
        let rotated = RotateHalf.eval(tvec![input.clone().into()])?;
        let mut back = args_1!(RotateHalf.eval(rotated)?).into_tensor();
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
