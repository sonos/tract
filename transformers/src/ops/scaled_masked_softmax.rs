use tract_nnef::internal::*;
use tract_nnef::tract_core::ops::binary::{BinMiniOp, TypedBinOp};
use tract_nnef::tract_core::ops::math::{Add, Mul};
use tract_nnef::tract_core::ops::nn::{Softmax, SoftmaxExp};

use crate::rule_ensure;

use super::{collect_node_const_inputs, previous_node, previous_nodes};

pub fn register(registry: &mut Registry) {
    registry.register_dumper(ser_scaled_masked_softmax);
    registry.register_primitive(
        "tract_transformers_scaled_masked_softmax",
        &[TypeName::Scalar.tensor().named("input"),
                  TypeName::Scalar.tensor().named("mask"),
                  TypeName::Scalar.named("scale")],
        &[("output", TypeName::Scalar.tensor())],
        de_scaled_masked_softmax,
    );
}

fn de_scaled_masked_softmax(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<Value> {
    let input = invocation.named_arg_as(builder, "input")?;
    let mask = invocation.named_arg_as(builder, "mask")?;
    let scale = invocation.named_arg_as(builder, "scale")?;
    builder.wire(ScaledMaskedSoftmax { scale }, &[input, mask])
}

fn ser_scaled_masked_softmax(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &ScaledMaskedSoftmax,
) -> TractResult<Option<Arc<RValue>>> {
    let input = ast.mapping[&node.inputs[0]].clone();
    let mask = ast.mapping[&node.inputs[1]].clone();
    Ok(Some(invocation(
        "tract_transformers_scaled_masked_softmax",
        &[input, mask],
        &[("scale", numeric(op.scale.cast_to_scalar::<f32>()?))],
    )))
}

/// A = SOFTMAX(INPUT * SCALE + MASK, AXIS=2)
/// Only input of rank of 3 is supported.
#[derive(Clone, Debug, Hash)]
pub struct ScaledMaskedSoftmax {
    pub scale: Arc<Tensor>,
}

impl Op for ScaledMaskedSoftmax {
    fn name(&self) -> Cow<str> {
        "ScaledMaskedSoftmax".to_string().into()
    }
    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("scale: {:?}", self.scale)])
    }
    op_as_typed_op!();
}

impl EvalOp for ScaledMaskedSoftmax {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let (input, mask) = args_2!(inputs);
        let dt = input.datum_type();
        let scale = self.scale.cast_to_dt(dt)?.into_owned();
        let scaled_input = Mul.eval(input, scale.into_tvalue(), dt)?;
        let masked_input = Add.eval(scaled_input.into(), mask, dt)?;
        let softmax = Softmax::new(tvec![2], None, SoftmaxExp::Libc)
            .eval(tvec![masked_input.into()])?[0]
            .clone();
        Ok(tvec![softmax])
    }
}

impl TypedOp for ScaledMaskedSoftmax {
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
    let Some(add_prev) = previous_node(model, node) else {
        return Ok(None);
    };
    let Some(add_prev_op) = add_prev.op_as::<TypedBinOp>() else {
        return Ok(None);
    };
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

    let Some(scale_op) = scale_node.op_as::<TypedBinOp>() else {
        return Ok(None);
    };
    rule_ensure!(scale_op.0.is::<Mul>());

    // Retrieve Scale
    let mul_consts = collect_node_const_inputs(model, scale_node);
    rule_ensure!(mul_consts.len() == 1);
    let scale = mul_consts[0].val().clone();

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
        ScaledMaskedSoftmax { scale },
        &[input, mask],
    )?;

    patch.shunt_outside(model, node.id.into(), out[0])?;
    Ok(Some(patch))
}
