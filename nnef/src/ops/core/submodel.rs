use tract_core::ops::submodel::SubmodelOp;

use crate::internal::*;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(ser_submodel);
    registry.register_primitive(
        "tract_core_submodel",
        &[TypeName::Scalar.tensor().array().named("input"), TypeName::String.named("label")],
        &[("outputs", TypeName::Any.tensor().array())],
        de_submodel,
    );
}

fn de_submodel(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let wires: TVec<OutletId> = invocation.named_arg_as(builder, "input")?;
    let label: String = invocation.named_arg_as(builder, "label")?;
    let model: TypedModel = builder
        .proto_model
        .resources
        .get(label.as_str())
        .with_context(|| anyhow!("{} not found in model builder loaded resources", label.as_str()))?
        .clone()
        .downcast_arc::<TypedModelResource>()
        .map_err(|_| anyhow!("Error while downcasting typed model resource"))
        .map(|r| r.0.clone())
        .with_context(|| anyhow!("Error while loading typed model resource"))?;

    let op: Box<dyn TypedOp> = Box::new(SubmodelOp::new(Box::new(model), &label)?);

    builder.model.wire_node(label, op, &wires).map(Value::from)
}

fn ser_submodel(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &SubmodelOp,
) -> TractResult<Option<Arc<RValue>>> {
    let input = tvec![ast.mapping[&node.inputs[0]].clone()];
    let invoke = invocation("tract_core_submodel", &input, &[("label", string(op.label()))]);
    ast.resources.insert(op.label().to_string(), Arc::new(TypedModelResource(op.model().clone())));
    Ok(Some(invoke))
}
