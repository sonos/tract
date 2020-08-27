use crate::internal::*;
use crate::ser::*;
use tract_core::ops;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(TypeId::of::<ops::array::MultiBroadcastTo>(), ser_broadcast);
    registry.register_primitive(
        "tract_core_broadcast",
        &[TypeName::Scalar.tensor().named("input"), TypeName::Integer.array().named("shape")],
        de_broadcast,
    );
}

fn de_broadcast(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<TVec<OutletId>> {
    let wire = invocation.named_arg_as(builder, "input")?;
    let shape = invocation.named_arg_as(builder, "shape")?;
    builder.wire(ops::array::MultiBroadcastTo { shape }, &[wire])
}

fn ser_broadcast(ast: &mut IntoAst, node: &TypedNode) -> TractResult<Option<Arc<RValue>>> {
    let op = node.op().downcast_ref::<ops::array::MultiBroadcastTo>().unwrap();
    let wire = ast.mapping[&node.inputs[0]].clone();
    let shape = op.shape.iter().map(|d| d.to_usize()).collect::<TractResult<TVec<usize>>>()?;
    Ok(Some(invocation("tract_core_broadcast", &[wire], &[("shape", ints(&shape))])))
}
