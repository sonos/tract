use crate::internal::*;
use crate::ser::*;
use tract_core::ops::math::{ComplexToInnerDim, InnerDimToComplex};

pub fn register(registry: &mut Registry) {
    registry.register_dumper(ser_ctid);
    registry.register_dumper(ser_idtc);
    registry.register_primitive(
        "tract_core_complex_to_inner_dim",
        &[TypeName::Complex.tensor().named("input")],
        &[("output", TypeName::Scalar.tensor())],
        de_ctid,
    );
    registry.register_primitive(
        "tract_core_inner_dim_to_complex",
        &[TypeName::Scalar.tensor().named("input")],
        &[("output", TypeName::Complex.tensor())],
        de_idtc,
    );
}

fn ser_ctid(
    ast: &mut IntoAst,
    node: &TypedNode,
    _: &ComplexToInnerDim,
) -> TractResult<Option<Arc<RValue>>> {
    let wire = ast.mapping[&node.inputs[0]].clone();
    Ok(Some(invocation("tract_core_complex_to_inner_dim", &[wire], &[])))
}

fn de_ctid(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let wire = invocation.named_arg_as(builder, "input")?;
    builder.wire(ComplexToInnerDim, &[wire])
}

fn ser_idtc(
    ast: &mut IntoAst,
    node: &TypedNode,
    _: &InnerDimToComplex,
) -> TractResult<Option<Arc<RValue>>> {
    let wire = ast.mapping[&node.inputs[0]].clone();
    Ok(Some(invocation("tract_core_inner_dim_to_complex", &[wire], &[])))
}

fn de_idtc(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let wire = invocation.named_arg_as(builder, "input")?;
    builder.wire(InnerDimToComplex, &[wire])
}
