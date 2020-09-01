use tract_nnef::internal::*;

element_wise_oop!(is_nan, IsNan,
    [f32] => bool |_, xs, ys| {
        xs.iter().zip(ys.iter_mut()).for_each(|(x,y)| *y = x.is_nan());
        Ok(())
    };
    prefix: "onnx."
);

pub fn lrn_parameters() -> Vec<Parameter> {
    vec![
        TypeName::Scalar.tensor().named("input"),
        TypeName::Scalar.named("alpha").default(0.0001),
        TypeName::Scalar.named("beta").default(0.75),
        TypeName::Scalar.named("bias").default(1.0),
        TypeName::Integer.named("size"),
    ]
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

