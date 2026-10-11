use crate::internal::*;
use crate::ser::*;
use tract_core::ops::array::CumSum;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(ser_cumsum);
    registry.register_primitive(
        "tract_core_cumsum",
        &[
            TypeName::Scalar.tensor().named("input"),
            TypeName::Integer.named("axis"),
            TypeName::Logical.named("exclusive"),
            TypeName::Logical.named("reverse"),
        ],
        &[("output", TypeName::Scalar.tensor())],
        de_cumsum,
    );
}

fn ser_cumsum(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &CumSum,
) -> TractResult<Option<Arc<RValue>>> {
    let input = ast.mapping[&node.inputs[0]].clone();
    Ok(Some(invocation(
        "tract_core_cumsum",
        &[input],
        &[
            ("axis", numeric(op.axis)),
            ("exclusive", logical(op.exclusive)),
            ("reverse", logical(op.reverse)),
        ],
    )))
}

fn de_cumsum(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let input = invocation.named_arg_as(builder, "input")?;
    let axis = invocation.named_arg_as(builder, "axis")?;
    let exclusive = invocation.named_arg_as(builder, "exclusive")?;
    let reverse = invocation.named_arg_as(builder, "reverse")?;
    builder.wire(CumSum { axis, exclusive, reverse }, &[input])
}
