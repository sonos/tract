use crate::rewrite_rules::next_node;
use crate::rewrite_rules::*;
use crate::rule_ensure;
use tract_core::internal::*;
use tract_core::ops::binary::TypedBinOp;
use tract_core::ops::element_wise::ElementWiseOp;
use tract_core::ops::math::{Mul, Pow, Tanh};

#[derive(Clone, Debug, Hash)]
/// NEW_GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)));
pub struct BasicNewGelu;

impl Op for BasicNewGelu {
    fn name(&self) -> Cow<str> {
        "BasicNewGelu".to_string().into()
    }
    op_as_typed_op!();
}

impl EvalOp for BasicNewGelu {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let dt = input.datum_type();

        let a_f32 = input.cast_to_dt(DatumType::F32)?;

        let sqrt_2_over_pi = (2.0 / std::f32::consts::PI).sqrt();

        let new_gelu_f32_data = a_f32
            .as_slice::<f32>()?
            .iter()
            .map(|x| 0.5 * x * (1.0 + f32::tanh(sqrt_2_over_pi * (x + 0.044715 * x.powi(3)))))
            .collect::<Vec<_>>();

        let new_gelu_f32 = Tensor::from_shape(input.shape(), &new_gelu_f32_data)?;
        Ok(tvec![new_gelu_f32.cast_to_dt(dt)?.into_owned().into_tvalue()])
    }
}

impl TypedOp for BasicNewGelu {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let dt = inputs[0].datum_type;
        let fact = dt.fact(inputs[0].shape.clone());
        Ok(tvec!(fact))
    }

    as_op!();
}

/// Search pattern => NEW_GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)));
pub fn as_new_gelu_rule(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    op: &TypedBinOp,
) -> TractResult<Option<TypedModelPatch>> {
    rule_ensure!(op.0.is::<Pow>());

    let pow_node = node;

    let in_fact = model.node_input_facts(pow_node.id)?[0];
    let dt = in_fact.datum_type;

    // Only F16 and F32 is supported.
    rule_ensure!(matches!(dt, DatumType::F32 | DatumType::F16));

    let mut patch = TypedModelPatch::default();
    let new_gelu_input = patch.taps(model, &pow_node.inputs)?;

    rule_ensure!(matches_single_input_const(model, pow_node, 3.0));

    // 0.044715 * x^3
    let Some(mul_coef_a) = find_succ_mul_with_const(model, pow_node, 0.044715) else {
        return Ok(None);
    };

    // x + 0.044715 * x^3
    let Some(x_plus_mul_coef_a) = find_succ_add_with(model, mul_coef_a, &pow_node.inputs[0]) else {
        return Ok(None);
    };

    // sqrt(2/pi) * (x + 0.044715 * x^3)
    let sqrt_2_over_pi = (2.0 / std::f32::consts::PI).sqrt();
    let Some(mul_sqrt_2_over_pi) =
        find_succ_mul_with_const(model, x_plus_mul_coef_a, sqrt_2_over_pi)
    else {
        return Ok(None);
    };

    // tanh(sqrt(2/pi) * (x + 0.044715 * x^3))
    let Some(tanh_succ) = next_node(model, mul_sqrt_2_over_pi) else { return Ok(None) };
    let Some(tanh_succ_op) = tanh_succ.op_as::<ElementWiseOp>() else { return Ok(None) };
    rule_ensure!(tanh_succ_op.0.is::<Tanh>());

    // 1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))
    let Some(tanh_plus_1) = find_succ_add_with_const(model, tanh_succ, 1.0) else {
        return Ok(None);
    };

    // Identify Mul
    let Some(mul_succ) = next_node(model, tanh_plus_1) else { return Ok(None) };
    let Some(mul_succ_op) = mul_succ.op_as::<TypedBinOp>() else { return Ok(None) };
    rule_ensure!(mul_succ_op.0.is::<Mul>());

    // Search first
    // tmp = x * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    // out = 0.5 * tmp
    let last_node_id = if mul_succ.inputs.contains(&pow_node.inputs[0]) {
        // 0.5 * x * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        let Some(last_mul_with_0_5) = find_succ_mul_with_const(model, mul_succ, 0.5) else {
            return Ok(None);
        };
        last_mul_with_0_5.id
    } else {
        // tmp = 0.5 * x
        // out = tmp * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        let Some(x_mul_0_5) = mul_succ
            .inputs
            .iter()
            .filter_map(|i| {
                let n = &model.nodes()[i.node];
                let op = n.op_as::<TypedBinOp>()?;
                op.0.is::<Mul>().then_some(n)
            })
            .next()
        else {
            return Ok(None);
        };
        rule_ensure!(matches_single_input_const(model, x_mul_0_5, 0.5));
        rule_ensure!(x_mul_0_5.inputs.contains(&pow_node.inputs[0]));
        mul_succ.id
    };

    let out =
        patch.wire_node(format!("{node_name}.new_gelu"), BasicNewGelu, &[new_gelu_input[0]])?;
    patch.shunt_outside(model, last_node_id.into(), out[0])?;
    Ok(Some(patch))
}
