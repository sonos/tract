use std::sync::Arc;

use crate::{
    internal::*,
    ser::{array, ints},
};
use tract_core::ops::force_eval::ForceEval;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(ser_force_eval);
    registry.register_primitive(
        "tract_core_force_eval",
        &[
            TypeName::Scalar.tensor().array().named("inputs"),
            TypeName::Integer.array().named("slots"),
        ],
        &[("output", TypeName::Scalar.tensor())],
        de_force_eval,
    );
}

fn ser_force_eval(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &ForceEval,
) -> TractResult<Option<Arc<RValue>>> {
    let wires: TVec<RValue> =
        node.inputs.iter().map(|it| ast.mapping[it].as_ref().clone()).collect();
    Ok(Some(invocation(
        "tract_core_force_eval",
        &[array(&wires).into()],
        &[("slots", ints(&op.slots))],
    )))
}

pub fn de_force_eval(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<Value> {
    let input_wires: TVec<OutletId> = invocation.named_arg_as(builder, "inputs")?;
    let output_slots: TVec<usize> = invocation.named_arg_as(builder, "slots")?;
    let force_eval_node = ForceEval::new(output_slots.to_vec());
    builder.wire(force_eval_node, &input_wires)
}
