use crate::rewrite_rules::{collect_node_const_inputs, previous_node, previous_nodes};
use crate::rule_ensure;
use tract_core::ops::binary::TypedBinOp;

use std::sync::Arc;
use tract_core::internal::*;
use tract_core::ops::binary::BinMiniOp;
use tract_core::ops::math::{Add, Mul};
use tract_core::ops::nn::{Softmax, SoftmaxExp};

/// A = SOFTMAX(INPUT * SCALE + MASK, AXIS=2)
/// Only input of rank of 3 is supported.
#[derive(Clone, Debug, Hash)]
pub struct BasicScaledMaskedSoftmax {
    pub scale: Arc<Tensor>,
}

impl Op for BasicScaledMaskedSoftmax {
    fn name(&self) -> Cow<str> {
        "BasicScaledMaskedSoftmax".to_string().into()
    }
    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("scale: {:?}", self.scale)])
    }
    op_as_typed_op!();
}

impl EvalOp for BasicScaledMaskedSoftmax {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let (input, mask) = args_2!(inputs);
        let dt = input.datum_type();
        let scaled_input = Mul.eval(input, self.scale.clone().into_tvalue(), dt)?;
        let masked_input = Add.eval(scaled_input.into(), mask, dt)?;
        let softmax = Softmax::new(tvec![2], None, SoftmaxExp::Libc)
            .eval(tvec![masked_input.into()])?[0]
            .clone();
        Ok(tvec![softmax])
    }
}

impl TypedOp for BasicScaledMaskedSoftmax {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.len() == 2);
        let (input, mask) = (inputs[0], inputs[1]);
        ensure!(input.datum_type == mask.datum_type);
        ensure!(input.rank() == 3 && mask.rank() == 3);
        let dt = input.datum_type;
        let fact = dt.fact(input.shape.clone());
        Ok(tvec!(fact))
    }

    as_op!();
}

/// Search pattern => A = SOFTMAX(A * SCALE + MASK, AXIS=2)
pub fn as_scaled_masked_softmax_rule(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    op: &Softmax,
) -> TractResult<Option<TypedModelPatch>> {
    rule_ensure!(op.axes.as_slice() == [2]);

    let in_fact = model.node_input_facts(node.id)?[0];
    let dt = in_fact.datum_type;
    // Only F16 and F32 is supported.
    rule_ensure!(matches!(dt, DatumType::F32 | DatumType::F16));

    // Identify Add operator (Mask)
    let Some(add_prev) = previous_node(model, node) else { return Ok(None) };
    let Some(add_prev_op) = add_prev.op_as::<TypedBinOp>() else { return Ok(None) };
    rule_ensure!(add_prev_op.0.is::<Add>());

    let mut in_add = previous_nodes(model, add_prev);
    rule_ensure!(in_add.len() == 2);

    in_add.reverse();
    let (left, right) = (in_add.pop().unwrap(), in_add.pop().unwrap());

    let (scale_node, mask_outlet) = if left.op_is::<TypedBinOp>() {
        (left, add_prev.inputs[1])
    } else {
        (right, add_prev.inputs[0])
    };

    let Some(scale_op) = scale_node.op_as::<TypedBinOp>() else { return Ok(None) };
    rule_ensure!(scale_op.0.is::<Mul>());

    // Retrieve Scale
    let mul_consts = collect_node_const_inputs(model, scale_node);
    rule_ensure!(mul_consts.len() == 1);
    let scale = mul_consts[0].0.clone();

    rule_ensure!(scale.len() == 1);
    rule_ensure!(scale.datum_type() == dt);

    // Ensure input and mask have the same rank
    rule_ensure!(model.outlet_fact(scale_node.inputs[0])?.shape.rank() == 3);
    rule_ensure!(model.outlet_fact(mask_outlet)?.shape.rank() == 3);

    let mut patch = TypedModelPatch::default();
    let input = patch.taps(model, &scale_node.inputs)?[0];
    let mask = patch.taps(model, &[mask_outlet])?[0];

    let out = patch.wire_node(
        format!("{node_name}.scaled_masked_softmax"),
        BasicScaledMaskedSoftmax { scale },
        &[input, mask],
    )?;

    patch.shunt_outside(model, node.id.into(), out[0])?;
    Ok(Some(patch))
}
