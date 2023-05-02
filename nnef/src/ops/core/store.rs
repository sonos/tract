use std::{any::TypeId, sync::Arc};

use crate::internal::*;
use tract_core::ops::store::Store;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(TypeId::of::<Store>(), ser_store);
    registry.register_primitive(
        "store",
        &[
            TypeName::Scalar.tensor().named("inputs"),
            TypeName::Scalar.tensor().named("state"),
            TypeName::String.named("id"),
        ],
        &[("output", TypeName::Scalar.tensor())],
        de_store,
    );
}

fn ser_store(ast: &mut IntoAst, node: &TypedNode) -> TractResult<Option<Arc<RValue>>> {
    let op = node.op().downcast_ref::<Store>().unwrap();
    let wires: TVec<Arc<RValue>> = node.inputs.iter().map(|it| ast.mapping[it].clone()).collect();
    Ok(Some(invocation("store", &wires, &[("id", string(op.id.clone()))])))
}

pub fn de_store(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let input_wire = invocation.named_arg_as(builder, "input")?;
    let state_wire = invocation.named_arg_as(builder, "state")?;
    let id: String = invocation.named_arg_as(builder, "id")?;
    let store_node = Store::new(&id);
    builder.wire(store_node, &[input_wire, state_wire])
}
