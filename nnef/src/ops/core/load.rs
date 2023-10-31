use std::sync::Arc;

use crate::internal::*;
use tract_core::ops::load::Load;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(ser_load);
    registry.register_primitive(
        "tract_core_load",
        &[TypeName::Scalar.tensor().array().named("input"), TypeName::String.named("id")],
        &[("output", TypeName::Scalar.tensor())],
        de_load,
    );
}

fn ser_load(ast: &mut IntoAst, node: &TypedNode, op: &Load) -> TractResult<Option<Arc<RValue>>> {
    let wire = ast.mapping[&node.inputs[0]].clone();
    Ok(Some(invocation("tract_core_load", &[wire], &[("id", string(op.id.clone()))])))
}

pub fn de_load(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let input_wire: OutletId = invocation.named_arg_as(builder, "input")?;
    let id: String = invocation.named_arg_as(builder, "id")?;
    let load_node = Load::new(&id);
    builder.wire(load_node, &[input_wire])
}
