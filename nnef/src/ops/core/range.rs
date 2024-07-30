use crate::internal::*;
use crate::ser::*;
use tract_core::ops::array::Range;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(range_dump);
    registry.register_primitive(
        "tract_core_range",
        &range_parameters(),
        &[("output", TypeName::Scalar.tensor())],
        range_load,
    );
}

fn range_parameters() -> Vec<Parameter> {
    vec![
        TypeName::Integer.named("start"),
        TypeName::Integer.named("end"),
        TypeName::Integer.named("step"),
    ]
}

fn range_dump(ast: &mut IntoAst, node: &TypedNode, _: &Range) -> TractResult<Option<Arc<RValue>>> {
    let start = ast.mapping[&node.inputs[0]].clone();
    let end = ast.mapping[&node.inputs[1]].clone();
    let step = ast.mapping[&node.inputs[2]].clone();

    Ok(Some(invocation("tract_core_range", &[start, end, step], &[])))
}

fn range_load(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let start: OutletId = invocation.named_arg_as(builder, "start")?;
    let end: OutletId = invocation.named_arg_as(builder, "end")?;
    let step: OutletId = invocation.named_arg_as(builder, "step")?;

    let len = builder.model.symbols.new_with_prefix("range");
    builder.wire(Range::new(len.into()), &[start, end, step])
}
