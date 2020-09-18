use crate::internal::*;
use crate::ser::*;
use tract_core::ops::cast::Cast;
use tract_core::ops::element_wise::ElementWiseOp;

pub fn register(registry: &mut Registry) {
    registry.register_element_wise(
        "tract_core_cast",
        TypeId::of::<tract_core::ops::cast::Cast>(),
        cast_dump,
        cast_parameters(),
        cast_load,
    );
}

fn cast_parameters() -> Vec<Parameter> {
    vec![TypeName::Scalar.tensor().named("input"), TypeName::String.named("to")]
}

fn cast_dump(ast: &mut IntoAst, node: &TypedNode) -> TractResult<Option<Arc<RValue>>> {
    let op = node.op_as::<ElementWiseOp>().unwrap().0.downcast_ref::<Cast>().unwrap();
    let input = ast.mapping[&node.inputs[0]].clone();
    Ok(Some(invocation(
        "tract_core_cast",
        &[input],
        &[("to", string(format!("{:?}", op.to).to_lowercase()))],
    )))
}

fn cast_load(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<TVec<OutletId>> {
    let input = invocation.named_arg_as(builder, "input")?;
    let to = invocation.named_arg_as::<String>(builder, "to")?.parse()?;
    builder.wire(ElementWiseOp(Box::new(Cast { to })), &[input])
}
