use crate::internal::*;

pub fn register(registry: &mut Registry) {
    registry.register_primitive(
        "tract_resource_get",
        &[
            TypeName::String.named("label").doc("Resource label to access"),
            TypeName::String.named("key").doc("Key path in resource"),
        ],
        &[("output", TypeName::Any.tensor())],
        resource_get,
    ).with_doc("Access embedded resource by key");
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
