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
    registry.register_primitive("tract_onnx_lrn", lrn_fragment(), lrn_load);
    registry
}

pub fn lrn_fragment() -> FragmentDecl {
    tract_nnef::ast::parse::parse_fragment_decl("fragment tract_onnx_lrn(input: tensor<scalar>, alpha: scalar = 0.0001, beta: scalar = 0.75, bias: scalar = 1.0, size: integer) -> (output: tensor<scalar>)").unwrap()
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
