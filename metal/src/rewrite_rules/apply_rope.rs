use crate::rewrite_rules::{previous_node, previous_nodes, single_prev_node_as, BasicRotateHalf};
use crate::rule_ensure;
use tract_core::internal::*;
use tract_core::ops::binary::BinMiniOp;
use tract_core::ops::binary::TypedBinOp;
use tract_core::ops::math::{Add, Mul};

#[derive(Clone, Debug, Hash)]
pub struct BasicApplyRope;

impl BasicApplyRope {
    pub fn is_supported_dt(dt: DatumType) -> bool {
        matches!(dt, DatumType::F32 | DatumType::F16)
    }
}

impl Op for BasicApplyRope {
    fn name(&self) -> Cow<str> {
        "BasicApplyRope".to_string().into()
    }
    op_as_typed_op!();
}

impl EvalOp for BasicApplyRope {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let (input, cos, sin) = args_3!(inputs);
        let rotated_input = args_1!(BasicRotateHalf.eval(tvec![input.clone()])?);
        let mul_with_cos = Mul.eval(input.clone(), cos, input.datum_type())?;
        let mul_with_sin = Mul.eval(rotated_input, sin, input.datum_type())?;
        let output = Add.eval(mul_with_cos.into(), mul_with_sin.into(), input.datum_type())?;
        Ok(tvec![output.into()])
    }
}

impl TypedOp for BasicApplyRope {
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
        single_prev_node_as::<BasicRotateHalf>(model, sin_mul)
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

    rule_ensure!(BasicApplyRope::is_supported_dt(model.outlet_fact(apply_rope_in)?.datum_type));
    rule_ensure!(BasicApplyRope::is_supported_dt(model.outlet_fact(cos)?.datum_type));
    rule_ensure!(BasicApplyRope::is_supported_dt(model.outlet_fact(sin)?.datum_type));

    let mut patch = TypedModelPatch::default();
    let input = patch.tap_model(model, apply_rope_in)?;
    let cos = patch.tap_model(model, cos)?;
    let sin = patch.tap_model(model, sin)?;
    let out =
        patch.wire_node(format!("{node_name}.apply_rope"), BasicApplyRope, &[input, cos, sin])?;
    patch.shunt_outside(model, node.id.into(), out[0])?;
    Ok(Some(patch))
}
