use crate::rewrite_rules::{collect_node_const_inputs, next_node};
use crate::rule_ensure;
use std::sync::Arc;
use tract_core::internal::*;
use tract_core::ops::binary::TypedBinOp;
use tract_core::ops::element_wise::ElementWiseOp;
use tract_core::ops::math::{Add, Mul, Rsqrt};
use tract_core::ops::nn::{Reduce, Reducer};

#[derive(Clone, Debug, Hash)]
pub struct BasicRmsNorm {
    pub axis: usize,
    pub eps: Arc<Tensor>,
}

impl Op for BasicRmsNorm {
    fn name(&self) -> Cow<str> {
        "BasicRmsNorm".to_string().into()
    }
    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axis: {:?}, eps: {:?}", self.axis, self.eps)])
    }
    op_as_typed_op!();
}

impl EvalOp for BasicRmsNorm {
    fn is_stateless(&self) -> bool {
        true
    }
}

impl TypedOp for BasicRmsNorm {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let dt = inputs[0].datum_type;
        let fact = dt.fact(inputs[0].shape.clone());
        Ok(tvec!(fact))
    }

    as_op!();
}

pub fn as_rms_norm_rule(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    op: &Reduce,
) -> TractResult<Option<TypedModelPatch>> {
    // Search pattern => A = A * RSQRT(MEAN_OF_SQUARES(A) + EPS);

    rule_ensure!(op.reducer == Reducer::MeanOfSquares);
    rule_ensure!(op.axes.len() == 1);
    let axis = op.axes[0];

    let in_fact = model.node_input_facts(node.id)?[0];
    let dt = in_fact.datum_type;

    // Only F16 and F32 is supported.
    rule_ensure!(matches!(dt, DatumType::F32 | DatumType::F16));

    let mut patch = TypedModelPatch::default();
    let rsm_input = patch.taps(model, &node.inputs)?;

    // Identify Add operator
    let Some(add_succ) = next_node(model, node) else { return Ok(None) };
    let Some(add_succ_op) = add_succ.op_as::<TypedBinOp>() else { return Ok(None) };
    rule_ensure!(add_succ_op.0.is::<Add>());

    // Retrieve epsilon
    let add_consts = collect_node_const_inputs(model, add_succ);
    rule_ensure!(add_consts.len() == 1);
    let eps = add_consts[0].0.clone();
    rule_ensure!(eps.len() == 1);
    rule_ensure!(eps.datum_type() == dt);

    // Identify Rsqrt
    let Some(rsqrt_succ) = next_node(model, add_succ) else { return Ok(None) };
    let Some(rsqrt_succ_op) = rsqrt_succ.op_as::<ElementWiseOp>() else { return Ok(None) };
    rule_ensure!(rsqrt_succ_op.0.is::<Rsqrt>());

    // Identify Mul
    let Some(mul_succ) = next_node(model, rsqrt_succ) else { return Ok(None) };
    let Some(mul_succ_op) = mul_succ.op_as::<TypedBinOp>() else { return Ok(None) };
    rule_ensure!(mul_succ_op.0.is::<Mul>());
    rule_ensure!(mul_succ.inputs.contains(&node.inputs[0]));

    let out =
        patch.wire_node(format!("{node_name}.rms_norm"), BasicRmsNorm { axis, eps }, &rsm_input)?;

    patch.shunt_outside(model, mul_succ.id.into(), out[0])?;

    Ok(Some(patch))
}
