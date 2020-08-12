use tract_hir::ops::element_wise::ElementWiseOp;
use tract_nnef::internal::*;

pub fn tract_nnef_onnx_registry() -> Registry {
    let mut registry: Registry = Registry::new("tract_onnx");
    macro_rules! dumper {
        ($op:ty, $path: path) => {
            registry.register_dumper(TypeId::of::<$op>(), |ast, node| {
                $path(ast, node, node.op().downcast_ref::<$op>().unwrap())
            })
        };
    };
    dumper!(crate::ops::nn::lrn::Lrn, lrn_dump);
    registry.register_primitive("tract_onnx_lrn", &lrn_parameters(), lrn_load);
    registry.register_element_wise(
        "tract_onnx_isinf",
        TypeId::of::<crate::ops::math::IsInf>(),
        isinf_dump,
        isinf_parameters(),
        isinf_load,
    );
    registry.register_unit_element_wise("tract_onnx_erf", &crate::ops::math::Erf {});
    registry.register_unit_element_wise("tract_onnx_is_nan", &crate::ops::math::IsNan {});
    registry
}

pub fn lrn_parameters() -> Vec<Parameter> {
    parse_parameters("input: tensor<scalar>, alpha: scalar = 0.0001, beta: scalar = 0.75, bias: scalar = 1.0, size: integer").unwrap()
}

pub fn lrn_dump(
    ast: &mut IntoAst,
    node: &TypedNode,
    lrn: &crate::ops::nn::lrn::Lrn,
) -> TractResult<Arc<RValue>> {
    let input = ast.mapping[&node.inputs[0]].clone();
    Ok(invocation(
        "tract_onnx_lrn",
        &[input],
        &[
            ("alpha", numeric(lrn.alpha)),
            ("beta", numeric(lrn.beta)),
            ("bias", numeric(lrn.bias)),
            ("size", numeric(lrn.size)),
        ],
    ))
}

pub fn lrn_load(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<TVec<OutletId>> {
    let input = invocation.named_arg_as(builder, "input")?;
    let alpha = invocation.named_arg_as(builder, "alpha")?;
    let beta = invocation.named_arg_as(builder, "beta")?;
    let bias = invocation.named_arg_as(builder, "bias")?;
    let size = invocation.named_arg_as(builder, "size")?;
    let op = crate::ops::nn::lrn::Lrn { alpha, beta, bias, size };
    builder.wire(op, &[input])
}

pub fn isinf_parameters() -> Vec<Parameter> {
    parse_parameters(
        "input: tensor<scalar>, detect_positive: logical = true, detect_negative: logical = true",
    )
    .unwrap()
}

pub fn isinf_dump(ast: &mut IntoAst, node: &TypedNode) -> TractResult<Arc<RValue>> {
    let op =
        node.op_as::<ElementWiseOp>().unwrap().0.downcast_ref::<crate::ops::math::IsInf>().unwrap();
    let input = ast.mapping[&node.inputs[0]].clone();
    Ok(invocation(
        "tract_onnx_isinf",
        &[input],
        &[
            ("detect_negative", logical(op.detect_negative)),
            ("detect_positive", logical(op.detect_positive)),
        ],
    ))
}

pub fn isinf_load(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<TVec<OutletId>> {
    let input = invocation.named_arg_as(builder, "input")?;
    let detect_positive = invocation.named_arg_as(builder, "detect_positive")?;
    let detect_negative = invocation.named_arg_as(builder, "detect_negative")?;
    let op = crate::ops::math::IsInf { detect_negative, detect_positive };
    builder.wire(ElementWiseOp(Box::new(op)), &[input])
}
