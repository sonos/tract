use tract_core::ops::element_wise::ElementWiseOp;
use tract_nnef::internal::*;

tract_core::element_wise_oop!(is_inf, IsInf { detect_positive: bool, detect_negative: bool },
    [f32] => bool |op, xs, ys| {
        xs.iter().zip(ys.iter_mut()).for_each(|(x,y)|
            *y = (op.detect_positive && *x == f32::INFINITY) || (op.detect_negative && *x == f32::NEG_INFINITY)
        );
        Ok(())
    },
    [f16] => bool |op, xs, ys| {
        xs.iter().zip(ys.iter_mut()).for_each(|(x,y)|
            *y = (op.detect_positive && *x == f16::INFINITY) || (op.detect_negative && *x == f16::NEG_INFINITY)
        );
        Ok(())
    };
    prefix: "onnx."
);

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
        "tract_onnx_isinf",
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
