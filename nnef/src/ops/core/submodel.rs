use tract_core::ops::submodel::SubmodelOp;

use crate::internal::*;

pub fn register(registry: &mut Registry) {
    registry.register_primitive(
        "tract_core_submodel",
        &[
            TypeName::String.named("name").default(""),
            TypeName::Scalar.tensor().array().named("input"),
            TypeName::String.named("label"),
        ],
        &[("outputs", TypeName::Any.tensor().array())],
        de_submodel,
    );
}

fn de_submodel(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let op_name = invocation.named_arg_as::<String>(builder, "name")?;
    let wires: TVec<OutletId> = invocation.named_arg_as(builder, "input")?;
    let label: String = invocation.named_arg_as(builder, "label")?;
    let model: TypedModel = builder
        .proto_model
        .resources
        .get(label.as_str())
        .with_context(|| {
            anyhow!(
                "{} not found in model builder loaded resources",
                label.as_str()
            )
        })?
        .clone()
        .downcast_arc::<TypedModelResource>()
        .map_err(|_| anyhow!("Error while downcasting typed model resource"))
        .and_then(|r| Ok(r.0.clone()))
        .with_context(|| anyhow!("Error while loading typed model resource"))?;

    let op: Box<dyn TypedOp> = Box::new(SubmodelOp::new(model)?);

    builder
        .model
        .wire_node(op_name, op, &wires)
        .map(Value::from)
}
