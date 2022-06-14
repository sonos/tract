use crate::internal::*;
use crate::ser::*;
use tract_core::ops::array::Range;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(TypeId::of::<Range>(), range_dump);
    registry.register_primitive("tract_core_range", &range_parameters(), range_load);
}

fn range_parameters() -> Vec<Parameter> {
    vec![
        TypeName::Scalar.tensor().named("start"),
        TypeName::Scalar.tensor().named("end"),
        TypeName::Scalar.tensor().named("step"),
    ]
}

fn range_dump(ast: &mut IntoAst, node: &TypedNode) -> TractResult<Option<Arc<RValue>>> {
    let op = node.op_as::<Range>().unwrap();
    let mut inputs = tvec![];

    inputs.push(ast.konst_variable(format!("{}_start", node.name), &Arc::new(op.start.clone()))?);
    inputs.push(ast.konst_variable(format!("{}_end", node.name), &Arc::new(op.end.clone()))?);
    inputs.push(ast.konst_variable(format!("{}_step", node.name), &Arc::new(op.step.clone()))?);

    Ok(Some(invocation("tract_core_range", &inputs, &[])))
}

fn range_load(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<TVec<OutletId>> {
    let start: Arc<Tensor> = invocation.named_arg_as(builder, "start")?;
    let end: Arc<Tensor> = invocation.named_arg_as(builder, "end")?;
    let step: Arc<Tensor> = invocation.named_arg_as(builder, "step")?;
    builder.wire(Range::new((*start).clone(), (*end).clone(), (*step).clone()), &[])
}
