use std::str::FromStr;

use crate::internal::*;
use tract_core::ops::einsum::EinSum;

pub fn register(registry: &mut Registry) {
    registry.register_primitive(
        "tract_core_qmatmul",
        &qmatmul_parameters(),
        &[("output", TypeName::Scalar.tensor())],
        qmatmul_load,
    );
}

fn qmatmul_parameters() -> Vec<Parameter> {
    vec![
        TypeName::Scalar.tensor().named("A"),
        TypeName::Scalar.tensor().named("B"),
        TypeName::Scalar.tensor().named("bias").default(0),
        TypeName::Integer.array().named("axes"),
        TypeName::Integer.spec().named("a0"),
        TypeName::Scalar.spec().named("a_scale"),
        TypeName::Integer.spec().named("b0"),
        TypeName::Scalar.spec().named("b_scale"),
        TypeName::Integer.spec().named("c0"),
        TypeName::Scalar.spec().named("c_scale"),
        TypeName::String.spec().named("output_type"),
    ]
}

pub fn qparams_as_outlets(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<TVec<OutletId>> {
    let a0: OutletId = invocation
        .named_arg_as::<OutletId>(builder, "a0")
        .or_else(|_| builder.add_const(rctensor0(0i32)))?;
    let a_scale: OutletId = invocation
        .named_arg_as::<OutletId>(builder, "a_scale")
        .or_else(|_| builder.add_const(rctensor0(1f32)))?;
    let b0: OutletId = invocation
        .named_arg_as::<OutletId>(builder, "b0")
        .or_else(|_| builder.add_const(rctensor0(0i32)))?;
    let b_scale: OutletId = invocation
        .named_arg_as::<OutletId>(builder, "b_scale")
        .or_else(|_| builder.add_const(rctensor0(1f32)))?;
    let c0: OutletId = invocation
        .named_arg_as::<OutletId>(builder, "c0")
        .or_else(|_| builder.add_const(rctensor0(0i32)))?;
    let c_scale: OutletId = invocation
        .named_arg_as::<OutletId>(builder, "c_scale")
        .or_else(|_| builder.add_const(rctensor0(1f32)))?;
    let a0 = builder.wire_as_outlets(tract_core::ops::cast::cast(i32::datum_type()), &[a0])?[0];
    let b0 = builder.wire_as_outlets(tract_core::ops::cast::cast(i32::datum_type()), &[b0])?[0];
    let c0 = builder.wire_as_outlets(tract_core::ops::cast::cast(i32::datum_type()), &[c0])?[0];
    Ok(tvec!(a0, a_scale, b0, b_scale, c0, c_scale))
}

fn qmatmul_load(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let a: OutletId = invocation.named_arg_as(builder, "A")?;
    let b: OutletId = invocation.named_arg_as(builder, "B")?;
    let bias: OutletId = invocation.named_arg_as(builder, "bias")?;
    let qparams = qparams_as_outlets(builder, invocation)?;
    let inputs: Vec<OutletId> = [a, b, bias].into_iter().chain(qparams).collect();
    let c_dt = if let Some(c) = invocation.dt_from_quant_file.first().cloned().flatten() {
        c
    } else {
        DatumType::from_str(&invocation.named_arg_as::<String>(builder, "output_type")?)?
    };
    let axes: TVec<usize> = invocation.named_arg_as(builder, "axes")?;
    let axes = from_legacy_axes_spec(&axes, builder.model.outlet_fact(a)?.rank())?;
    builder.wire(EinSum { axes, operating_dt: i32::datum_type(), q_params: Some(c_dt) }, &inputs)
}

pub fn from_legacy_axes_spec(spec: &[usize], rank: usize) -> TractResult<AxesMapping> {
    let [a_m, a_k, b_k, b_n, c_m, c_n] = spec else { bail!("Invalid axes specification")};
    AxesMapping::disconnected_for_ranks(&[rank, rank], &[rank])?
        .renaming((InOut::In(0), *a_m), 'm')?
        .linking('m', (InOut::Out(0), *c_m))?
        .renaming((InOut::In(0), *a_k), 'k')?
        .linking('k', (InOut::In(1), *b_k))?
        .renaming((InOut::In(1), *b_n), 'n')?
        .linking('n', (InOut::In(0), *c_n))
}
