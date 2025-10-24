use tract_nnef::internal::*;
use tract_nnef::tract_core::ops::binary::{BinMiniOp, TypedBinOp};
use tract_nnef::tract_core::ops::element_wise::ElementWiseOp;
use tract_nnef::tract_core::ops::math::{Add, Mul, Rsqrt};
use tract_nnef::tract_core::ops::nn::{Reduce, Reducer};

use crate::rule_ensure;

use super::{collect_node_const_inputs, next_node};

pub fn register(registry: &mut Registry) {
    registry.register_dumper(ser_rms_norm);
    registry.register_primitive(
        "tract_transformers_rms_norm",
        &[
            TypeName::Scalar.tensor().named("input"),
            TypeName::Integer.named("axis"),
            TypeName::Scalar.named("eps").default(1e-6f32),
        ],
        &[("output", TypeName::Scalar.tensor())],
        de_rms_norm,
    );
}

fn de_rms_norm(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let input = invocation.named_arg_as(builder, "input")?;
    let axis: usize = invocation.named_arg_as(builder, "axis")?;
    let eps = invocation.named_arg_as(builder, "eps")?;
    builder.wire(RmsNorm { axis, eps }, &[input])
}

fn ser_rms_norm(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &RmsNorm,
) -> TractResult<Option<Arc<RValue>>> {
    let input = ast.mapping[&node.inputs[0]].clone();
    Ok(Some(invocation(
        "tract_transformers_rms_norm",
        &[input],
        &[("axis", numeric(op.axis)), ("eps", numeric(op.eps.cast_to_scalar::<f32>()?))],
    )))
}

#[derive(Clone, Debug, Hash)]
pub struct RmsNorm {
    pub axis: usize,
    pub eps: Arc<Tensor>,
}

impl Op for RmsNorm {
    fn name(&self) -> StaticName {
        "RmsNorm".to_string().into()
    }
    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axis: {:?}, eps: {:?}", self.axis, self.eps)])
    }
    op_as_typed_op!();
}

impl EvalOp for RmsNorm {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);

        let input_f32 = input.cast_to::<f32>()?.into_owned();
        let a1 = Reducer::MeanOfSquares.reduce(&[self.axis], &input_f32)?;
        let mut a2 = Add.eval(a1.into_tvalue(), self.eps.clone().into_tvalue(), DatumType::F32)?;
        Rsqrt {}.eval_in_place(&mut a2, None)?;
        let a3 = Mul.eval(a2.into_tvalue(), input_f32.into_tvalue(), DatumType::F32)?;
        Ok(tvec![a3.cast_to_dt(input.datum_type())?.into_owned().into()])
    }
}

impl TypedOp for RmsNorm {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let dt = inputs[0].datum_type;
        let fact = dt.fact(inputs[0].shape.clone());
        Ok(tvec!(fact))
    }

    as_op!();
}

/// Search pattern => A = A * RSQRT(MEAN_OF_SQUARES(A) + EPS)
pub fn rms_norm_rule(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    op: &Reduce,
) -> TractResult<Option<TypedModelPatch>> {
    rule_ensure!(op.reducer == Reducer::MeanOfSquares);
    rule_ensure!(op.axes.len() == 1);
    let axis = op.axes[0];

    let in_fact = model.node_input_facts(node.id)?[0];
    let dt = in_fact.datum_type;

    // Only F16 and F32 is supported.
    rule_ensure!(matches!(dt, DatumType::F32 | DatumType::F16));

    // Identify Add operator
    let Some(add_succ) = next_node(model, node) else {
        return Ok(None);
    };
    let Some(add_succ_op) = add_succ.op_as::<TypedBinOp>() else {
        return Ok(None);
    };
    rule_ensure!(add_succ_op.0.is::<Add>());

    // Retrieve epsilon
    let add_consts = collect_node_const_inputs(model, add_succ);
    rule_ensure!(add_consts.len() == 1);
    let eps = add_consts[0].val().clone();
    rule_ensure!(eps.len() == 1);
    rule_ensure!(eps.datum_type() == dt);

    // Identify Rsqrt
    let Some(rsqrt_succ) = next_node(model, add_succ) else {
        return Ok(None);
    };
    let Some(rsqrt_succ_op) = rsqrt_succ.op_as::<ElementWiseOp>() else {
        return Ok(None);
    };
    rule_ensure!(rsqrt_succ_op.0.is::<Rsqrt>());

    // Identify Mul
    let Some(mul_succ) = next_node(model, rsqrt_succ) else {
        return Ok(None);
    };
    let Some(mul_succ_op) = mul_succ.op_as::<TypedBinOp>() else {
        return Ok(None);
    };
    rule_ensure!(mul_succ_op.0.is::<Mul>());
    rule_ensure!(mul_succ.inputs.contains(&node.inputs[0]));

    let mut patch = TypedModelPatch::default();
    let rsm_input = patch.taps(model, &node.inputs)?;
    let out =
        patch.wire_node(format!("{node_name}.rms_norm"), RmsNorm { axis, eps }, &rsm_input)?;

    patch.shunt_outside(model, mul_succ.id.into(), out[0])?;
    Ok(Some(patch))
}
