use crate::internal::*;
use crate::ser::*;
use tract_core::ops::array::Range;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(TypeId::of::<Range>(), range_dump);
    registry.register_primitive("tract_core_range", &range_parameters(), range_load);
}

fn range_parameters() -> Vec<Parameter> {
    vec![
        TypeName::Integer.named("start"),
        TypeName::Integer.named("end"),
        TypeName::Integer.named("step"),
    ]
}

fn range_dump(_ast: &mut IntoAst, node: &TypedNode) -> TractResult<Option<Arc<RValue>>> {
    let op = node.op_as::<Range>().unwrap();

    let start = op.start.to_scalar::<TDim>()?;
    let end = op.end.to_scalar::<TDim>()?;
    let step = op.step.to_scalar::<TDim>()?;

    Ok(Some(invocation(
        "tract_core_range",
        &[],
        &[("start", tdim(start)), ("end", tdim(end)), ("step", tdim(step))],
    )))
}

fn range_load(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<TVec<OutletId>> {
    let start: TDim = invocation.named_arg_as(builder, "start")?;
    let end: TDim = invocation.named_arg_as(builder, "end")?;
    let step: TDim = invocation.named_arg_as(builder, "step")?;

    let start: Tensor = start.into();
    let end: Tensor = end.into();
    let step: Tensor = step.into();

    builder.wire(Range::new(start, end, step), &[])
}
