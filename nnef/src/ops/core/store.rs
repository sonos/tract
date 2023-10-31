use std::sync::Arc;

use crate::internal::*;
use tract_core::ops::store::Store;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(ser_store);
    registry.register_primitive(
        "tract_core_store",
        &[
            TypeName::Scalar.tensor().named("input"),
            TypeName::Scalar.tensor().named("state"),
            TypeName::String.named("id"),
        ],
        &[("output", TypeName::Scalar.tensor())],
        de_store,
    );
}

fn ser_store(ast: &mut IntoAst, node: &TypedNode, op: &Store) -> TractResult<Option<Arc<RValue>>> {
    let wires: TVec<RValue> = node.inputs.iter().map(|it| (*ast.mapping[it]).clone()).collect();
    Ok(Some(invocation(
        "tract_core_store",
        &[],
        &[("input", wires[0].clone()), ("state", wires[1].clone()), ("id", string(op.id.clone()))],
    )))
}

pub fn de_store(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let input_wire = invocation.named_arg_as(builder, "input")?;
    let state_wire = invocation.named_arg_as(builder, "state")?;
    let id: String = invocation.named_arg_as(builder, "id")?;
    let store_node = Store::new(&id);
    builder.wire(store_node, &[input_wire, state_wire])
}
