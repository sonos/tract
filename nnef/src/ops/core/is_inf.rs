use crate::internal::*;
use crate::ser::*;
use tract_core::ops::element_wise::ElementWiseOp;
use tract_core::ops::math::IsInf;

pub fn parameters() -> Vec<Parameter> {
    vec![
        TypeName::Scalar.tensor().named("input"),
        TypeName::Logical.named("detect_positive").default(true),
        TypeName::Logical.named("detect_negative").default(true),
    ]
}

pub fn dump(ast: &mut IntoAst, node: &TypedNode) -> TractResult<Option<Arc<RValue>>> {
    let op = node.op_as::<ElementWiseOp>().unwrap().0.downcast_ref::<IsInf>().unwrap();
    let input = ast.mapping[&node.inputs[0]].clone();
    Ok(Some(invocation(
        "tract_core_is_inf",
        &[input],
        &[
            ("detect_negative", logical(op.detect_negative)),
            ("detect_positive", logical(op.detect_positive)),
        ],
    )))
}

pub fn load(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let input = invocation.named_arg_as(builder, "input")?;
    let detect_positive = invocation.named_arg_as(builder, "detect_positive")?;
    let detect_negative = invocation.named_arg_as(builder, "detect_negative")?;
    let op = IsInf { detect_negative, detect_positive };
    builder.wire(ElementWiseOp(Box::new(op), None), &[input])
}

pub fn register(registry: &mut Registry) {
    registry.register_element_wise(
        "tract_core_is_inf",
        TypeId::of::<IsInf>(),
        Box::new(dump),
        parameters(),
        load,
    );
}
