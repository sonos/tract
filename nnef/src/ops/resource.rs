use crate::internal::*;

pub fn register(registry: &mut Registry) {
    registry.register_primitive(
        "tract_resource_get",
        &[
            TypeName::String.named("label"),
            TypeName::String.named("key"),
        ],
        resource_get,
    );
}

fn resource_get(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<Value> {
    let label: String = invocation.named_arg_as(builder, "label")?;
    let key: String = invocation.named_arg_as(builder, "key")?;
    let resource = builder.proto_model.resources.get(&label)
        .with_context(|| anyhow!("No resource found for label {:?} in the model", label))?;
    resource.get(&key)
}