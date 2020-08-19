use crate::internal::*;
use crate::ser::*;
use tract_core::ops;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(TypeId::of::<ops::array::Gather>(), ser_gather);
    registry.register_primitive(
        "tract_core_gather",
        &[
            TypeName::Scalar.tensor().named("input"),
            TypeName::Scalar.tensor().named("indices"),
            TypeName::Integer.named("axis"),
        ],
        de_gather,
    );
}

fn ser_gather(ast: &mut IntoAst, node: &TypedNode) -> TractResult<Option<Arc<RValue>>> {
    let op = node.op().downcast_ref::<ops::array::Gather>().unwrap();
    let wire = ast.mapping[&node.inputs[0]].clone();
    let indices = ast.mapping[&node.inputs[1]].clone();
    Ok(Some(invocation("tract_core_gather", &[wire, indices], &[("axis", numeric(op.axis))])))
}

fn de_gather(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<TVec<OutletId>> {
    let wire = invocation.named_arg_as(builder, "input")?;
    let indices = invocation.named_arg_as(builder, "indices")?;
    let axis = invocation.named_arg_as(builder, "axis")?;
    builder.wire(ops::array::Gather { axis }, &[wire, indices])
}
