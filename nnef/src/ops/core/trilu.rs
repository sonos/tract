use crate::internal::*;
use crate::ser::*;
use tract_core::ops::array::Trilu;
use tract_core::ops::cast::cast;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(ser_trilu);
    registry.register_primitive(
        "tract_core_trilu",
        &[
            TypeName::Scalar.tensor().named("input"),
            TypeName::Integer.tensor().named("k"),
            TypeName::Logical.named("upper"),
        ],
        &[("output", TypeName::Scalar.tensor())],
        de_trilu,
    );
}

fn ser_trilu(ast: &mut IntoAst, node: &TypedNode, op: &Trilu) -> TractResult<Option<Arc<RValue>>> {
    let input = ast.mapping[&node.inputs[0]].clone();
    let k = ast.mapping[&node.inputs[1]].clone();
    Ok(Some(invocation("tract_core_trilu", &[input, k], &[("upper", logical(op.upper))])))
}

fn de_trilu(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let input = invocation.named_arg_as(builder, "input")?;
    let k = invocation.named_arg_as(builder, "k")?;
    let upper = invocation.named_arg_as(builder, "upper")?;
    let k_casted = builder.wire_as_outlets(cast(DatumType::I64), &[k])?[0];
    builder.wire(Trilu { upper }, &[input, k_casted])
}
