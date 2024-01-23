use tract_core::ops::nn::{Softmax, SoftmaxExp};

use crate::internal::*;

pub fn register(registry: &mut Registry) {
    registry.register_primitive(
        "tract_core_softmax",
        &[
            TypeName::Scalar.tensor().named("x"),
            TypeName::Integer.tensor().named("axes"),
            TypeName::String.named("exp"),
        ],
        &[("output", TypeName::Scalar.tensor())],
        softmax,
    );
}

pub fn softmax(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let x = invocation.named_arg_as(builder, "x")?;
    let axes: TVec<usize> = invocation.named_arg_as(builder, "axes")?;

    let input_fact = builder.model.outlet_fact(x)?.clone();
    let quant_output_dt = if input_fact.datum_type.is_float() {
        None
    } else {
        invocation.dt_from_quant_file.first().cloned().flatten()
    };

    let exp: Option<String> = invocation.get_named_arg_as(builder, "exp")?;
    let exp = match exp.as_deref() {
        Some("fast_compact") => SoftmaxExp::FastCompact,
        _ => SoftmaxExp::Libc
    };

    builder.wire(Softmax { axes, quant_output_dt, exp }, &[x])
}
