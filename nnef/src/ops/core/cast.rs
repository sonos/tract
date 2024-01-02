use crate::internal::*;
use crate::ser::*;
use tract_core::ops::cast::Cast;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(cast_dump);
    registry.register_primitive(
        "tract_core_cast",
        &cast_parameters(),
        &[("output", TypeName::Scalar.tensor())],
        cast_load,
    );
}

fn cast_parameters() -> Vec<Parameter> {
    vec![TypeName::Scalar.tensor().named("input"), TypeName::String.named("to")]
}

fn cast_dump(ast: &mut IntoAst, node: &TypedNode, op: &Cast) -> TractResult<Option<Arc<RValue>>> {
    let input = ast.mapping[&node.inputs[0]].clone();
    Ok(Some(invocation("tract_core_cast", &[input], &[("to", datum_type(op.to))])))
}

fn cast_load(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let input = invocation.named_arg_as(builder, "input")?;
    let invocation_dt = invocation.dt_from_quant_file.first().copied().flatten();
    let to = if let Ok(s) = invocation.named_arg_as::<String>(builder, "to") {
        let dt: DatumType = s.parse()?;
        if let Some(invocation_dt) = invocation_dt {
            if invocation_dt.unquantized() != dt.unquantized() {
                bail!("Mismatched cast: graph.quant {:?}, got graph.nnef {:?}", invocation_dt, dt)
            } else {
                invocation_dt
            }
        } else {
            dt
        }
    } else {
        invocation_dt.context("No datum type for cast")?
    };
    builder.wire(Cast { to }, &[input])
}
