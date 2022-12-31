use crate::internal::*;
use crate::ser::*;
use tract_core::ops;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(TypeId::of::<ops::math::ComplexToInnerDim>(), ser_ctid);
    registry.register_dumper(TypeId::of::<ops::math::InnerDimToComplex>(), ser_idtc);
    registry.register_primitive(
        "tract_core_complex_to_inner_dim",
        &[TypeName::Scalar.tensor().named("input")],
        &[("output", TypeName::Scalar.tensor())],
        de_ctid,
    );
    registry.register_primitive(
        "tract_core_inner_dim_to_complex",
        &[TypeName::Scalar.tensor().named("input")],
        &[("output", TypeName::Scalar.tensor())],
        de_idtc,
    );
}

fn ser_ctid(ast: &mut IntoAst, node: &TypedNode) -> TractResult<Option<Arc<RValue>>> {
    let op = node.op().downcast_ref::<ops::math::ComplexToInnerDim>().unwrap();
    let wire = ast.mapping[&node.inputs[0]].clone();
    Ok(Some(invocation("tract_core_complex_to_inner_dim", &[wire], &[])))
}

fn de_ctid(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let wire = invocation.named_arg_as(builder, "input")?;
    builder.wire(ops::math::ComplexToInnerDim, &[wire])
}

fn ser_idtc(ast: &mut IntoAst, node: &TypedNode) -> TractResult<Option<Arc<RValue>>> {
    let op = node.op().downcast_ref::<ops::math::InnerDimToComplex>().unwrap();
    let wire = ast.mapping[&node.inputs[0]].clone();
    Ok(Some(invocation("tract_core_inner_dim_to_complex", &[wire], &[])))
}

fn de_idtc(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let wire = invocation.named_arg_as(builder, "input")?;
    builder.wire(ops::math::InnerDimToComplex, &[wire])
}
