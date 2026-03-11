use crate::internal::*;
use crate::ops::element_wise::ElementWiseOp;
use crate::ops::math::Mul;
use crate::ops::nn::Sigmoid;

use tract_data::half::f16;

element_wise!(silu, Silu,
    [f16] => |_, xs| {
        xs.iter_mut().for_each(|x| {
            let xf = x.to_f32();
            *x = f16::from_f32(xf / (1.0 + (-xf).exp()));
        });
        Ok(())
    },
    [f32] => |_, xs| {
        let mut sigmoid = xs.to_vec();
        (tract_linalg::ops().sigmoid_f32)().run(&mut sigmoid)?;
        xs.iter_mut().zip(sigmoid).for_each(|(x, s)| *x *= s);
        Ok(())
    };
    declutter: detect_silu
);

/// Search pattern => A = A * SIGMOID(A)
pub fn detect_silu(model: &TypedModel, node: &TypedNode) -> TractResult<Option<TypedModelPatch>> {
    rule_if!(node.op_as::<ElementWiseOp>().is_some_and(|op| op.0.is::<Sigmoid>()));

    let in_fact = model.node_input_facts(node.id)?[0];
    let dt = in_fact.datum_type;

    // Only F16 and F32 is supported.
    rule_if!(matches!(dt, DatumType::F32 | DatumType::F16));

    // Identify Mul successor: Sigmoid(A) * A
    rule_if_some!(mul_succ = model.find_succ_bin_with_outlet::<Mul>(node, &node.inputs[0]));

    let mut patch = TypedModelPatch::default();
    let silu_input = patch.taps(model, &node.inputs)?;
    let out = patch.wire_node(format!("{}.silu", node.name), silu(), &silu_input)?;
    patch.shunt_outside(model, mul_succ.id.into(), out[0])?;
    Ok(Some(patch))
}
