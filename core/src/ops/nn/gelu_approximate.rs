use crate::internal::*;
use crate::ops::binary::TypedBinOp;
use crate::ops::element_wise::ElementWiseOp;
use crate::ops::math::{Add, Mul, Pow, Tanh};

use tract_data::half::f16;

fn gelu_approx_f32(x: f32, pow: i32) -> f32 {
    let sqrt_2_over_pi = (2.0 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + f32::tanh(sqrt_2_over_pi * (x + 0.044715 * x.powi(pow))))
}

element_wise!(gelu_approximate, GeluApproximate { fast_impl: bool },
    [f16] => |op, xs| {
        let pow = if op.fast_impl { 2 } else { 3 };
        xs.iter_mut().for_each(|x| {
            *x = f16::from_f32(gelu_approx_f32(x.to_f32(), pow));
        });
        Ok(())
    },
    [f32] => |op, xs| {
        let pow = if op.fast_impl { 2 } else { 3 };
        xs.iter_mut().for_each(|x| {
            *x = gelu_approx_f32(*x, pow);
        });
        Ok(())
    }
);

/// Search pattern => NEW_GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^N))); N ∈ {2, 3}
pub fn detect_gelu_approx(
    _op: &Pow,
    model: &TypedModel,
    node: &TypedNode,
) -> TractResult<Option<TypedModelPatch>> {
    let pow_node = node;

    let in_fact = model.node_input_facts(pow_node.id)?[0];
    let dt = in_fact.datum_type;

    // Only F16 and F32 is supported.
    rule_if!(matches!(dt, DatumType::F32 | DatumType::F16));

    rule_if!(
        model.matches_single_input_const(pow_node, 3.0)
            || model.matches_single_input_const(pow_node, 2.0)
    );
    let fast_impl = model.matches_single_input_const(pow_node, 2.0);

    // 0.044715 * x^N
    rule_if_some!(mul_coef_a = model.find_succ_bin_with_const::<Mul>(pow_node, 0.044715));

    // x + 0.044715 * x^N
    rule_if_some!(
        x_plus_mul_coef_a = model.find_succ_bin_with_outlet::<Add>(mul_coef_a, &pow_node.inputs[0])
    );

    // sqrt(2/pi) * (x + 0.044715 * x^N)
    let sqrt_2_over_pi = (2.0 / std::f32::consts::PI).sqrt();
    rule_if_some!(
        mul_sqrt_2_over_pi =
            model.find_succ_bin_with_const::<Mul>(x_plus_mul_coef_a, sqrt_2_over_pi)
    );

    // tanh(sqrt(2/pi) * (x + 0.044715 * x^N))
    rule_if_some!(tanh_succ = model.single_succ(mul_sqrt_2_over_pi.id)?);
    rule_if_some!(tanh_succ_op = tanh_succ.op_as::<ElementWiseOp>());
    rule_if!(tanh_succ_op.0.is::<Tanh>());

    // 1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^N)) N ∈ {2, 3}
    rule_if_some!(tanh_plus_1 = model.find_succ_bin_with_const::<Add>(tanh_succ, 1.0));

    // Identify Mul
    rule_if_some!(mul_succ = model.single_succ(tanh_plus_1.id)?);
    rule_if_some!(mul_succ_op = mul_succ.op_as::<TypedBinOp>());
    rule_if!(mul_succ_op.0.is::<Mul>());

    // Search first
    // tmp = x * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^N)))
    // out = 0.5 * tmp
    let last_node_id = if mul_succ.inputs.contains(&pow_node.inputs[0]) {
        // 0.5 * x * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^N)))
        rule_if_some!(last_mul_with_0_5 = model.find_succ_bin_with_const::<Mul>(mul_succ, 0.5));
        last_mul_with_0_5.id
    } else {
        // tmp = 0.5 * x
        // out = tmp * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^N))) N ∈ {2, 3}
        rule_if_some!(
            x_mul_0_5 = mul_succ
                .inputs
                .iter()
                .filter_map(|i| {
                    let n = &model.nodes()[i.node];
                    let op = n.op_as::<TypedBinOp>()?;
                    op.0.is::<Mul>().then_some(n)
                })
                .next()
        );
        rule_if!(model.matches_single_input_const(x_mul_0_5, 0.5));
        rule_if!(x_mul_0_5.inputs.contains(&pow_node.inputs[0]));
        mul_succ.id
    };

    let mut patch = TypedModelPatch::default();
    let gelu_approx_input = patch.taps(model, &pow_node.inputs)?;
    let out = patch.wire_node(
        format!("{}.gelu_approx", pow_node.name),
        gelu_approximate(fast_impl),
        &[gelu_approx_input[0]],
    )?;
    patch.shunt_outside(model, last_node_id.into(), out[0])?;
    Ok(Some(patch))
}
