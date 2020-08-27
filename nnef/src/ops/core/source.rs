use crate::internal::*;
use crate::ser::*;
use tract_core::ops::source::TypedSource;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(TypeId::of::<TypedSource>(), external_dump);
    registry.register_primitive("tract_core_external", &external_parameters(), external_load);
}

fn external_dump(_ast: &mut IntoAst, node: &TypedNode) -> TractResult<Option<Arc<RValue>>> {
    let op = node.op_as::<TypedSource>().unwrap();
    Ok(Some(invocation(
        "tract_core_external",
        &[],
        &[
            ("shape", ints(&op.fact.shape.as_finite().unwrap())),
            ("datum_type", string(format!("{:?}", op.fact.datum_type))),
        ],
    )))
}

fn external_parameters() -> Vec<Parameter> {
    vec![TypeName::String.named("datum_type"), TypeName::Integer.array().named("shape")]
}

fn external_load(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<TVec<OutletId>> {
    let shape: TVec<usize> = invocation.named_arg_as(builder, "shape")?;
    let dt = invocation.named_arg_as::<String>(builder, "datum_type")?.parse()?;
    let fact = TypedFact::dt_shape(dt, &*shape)?;
    Ok(tvec!(builder.model.add_source("", fact)?))
}
