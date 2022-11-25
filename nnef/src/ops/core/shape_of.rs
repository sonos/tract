use crate::internal::*;

pub fn register(registry: &mut Registry) {
    // No serialization is done since: operation follow ONNX design:
    // At deserialization we wire it to a constant used by tract.
    // This make the operation serialization/deserialization non-symmetric
    registry.register_primitive(
        "tract_core_shape_of",
        &[TypeName::Scalar.tensor().named("input")],
        &[("output", TypeName::Integer.tensor())],
        de_shape_of,
    );
}

fn de_shape_of(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<Value> {
    let input = invocation.named_arg_as(builder, "input")?;
    let shape = tensor1(&builder.model.outlet_fact(input)?.shape.to_tvec());
    let wire = builder.model.add_const("shape", shape)?;
    Ok(Value::Wire(wire))
}
