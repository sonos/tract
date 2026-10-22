use crate::internal::*;
use crate::ser::ints;
use tract_core::ops::nn::GroupNorm;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(ser_group_norm);
    registry.register_primitive(
        "tract_core_group_norm",
        &[
            TypeName::Scalar.tensor().named("input"),
            TypeName::Integer.named("axis"),
            TypeName::Integer.named("groups"),
            TypeName::Integer.array().named("reduced_axes"),
            TypeName::Scalar.named("eps").default(1e-5f32),
        ],
        &[("output", TypeName::Scalar.tensor())],
        de_group_norm,
    );
}

fn de_group_norm(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<Value> {
    let input = invocation.named_arg_as(builder, "input")?;
    let axis: usize = invocation.named_arg_as(builder, "axis")?;
    let groups: usize = invocation.named_arg_as(builder, "groups")?;
    let reduced_axes: TVec<usize> = invocation.named_arg_as(builder, "reduced_axes")?;
    let eps = invocation.named_arg_as(builder, "eps")?;
    builder.wire(GroupNorm { axis, groups, reduced_axes, eps }, &[input])
}

fn ser_group_norm(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &GroupNorm,
) -> TractResult<Option<Arc<RValue>>> {
    let input = ast.mapping[&node.inputs[0]].clone();
    Ok(Some(invocation(
        "tract_core_group_norm",
        &[input],
        &[
            ("axis", numeric(op.axis)),
            ("groups", numeric(op.groups)),
            ("reduced_axes", ints(&op.reduced_axes)),
            ("eps", numeric(op.eps.cast_to_scalar::<f32>()?)),
        ],
    )))
}
