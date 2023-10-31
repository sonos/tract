use crate::internal::*;
use crate::ser::*;
use tract_core::ops::source::TypedSource;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(external_dump);
    registry.register_primitive(
        "tract_core_external",
        &external_parameters(),
        &[("output", TypeName::Any.tensor())],
        external_load,
    );
}

fn external_dump(
    ast: &mut IntoAst,
    _node: &TypedNode,
    op: &TypedSource,
) -> TractResult<Option<Arc<RValue>>> {
    for dim in op.fact.shape.iter() {
        for sym in dim.symbols() {
            ast.ensure_symbol(&sym)?;
        }
    }
    let shape = tdims(&op.fact.shape);
    Ok(Some(invocation(
        "tract_core_external",
        &[],
        &[
            ("shape", shape),
            ("datum_type", string(format!("{:?}", op.fact.datum_type.unquantized()))),
        ],
    )))
}

fn external_parameters() -> Vec<Parameter> {
    vec![TypeName::String.named("datum_type"), TypeName::Integer.array().named("shape")]
}

fn external_load(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<Value> {
    let shape: TVec<TDim> =
        builder.allowing_new_symbols(|builder| invocation.named_arg_as(builder, "shape"))?;
    let mut dt: DatumType = invocation.named_arg_as::<String>(builder, "datum_type")?.parse()?;
    if let Some(Some(qdt)) = invocation.dt_from_quant_file.get(0) {
        dt = *qdt;
    }
    Ok(Value::Wire(builder.model.add_source("", dt.fact(&*shape))?))
}
