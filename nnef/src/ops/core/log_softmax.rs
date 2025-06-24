use tract_core::ops::nn::{Softmax, SoftmaxKind};

use crate::internal::*;

pub fn register(registry: &mut Registry) {
    registry.register_primitive(
        "tract_core_log_softmax",
        &[
            TypeName::Scalar.tensor().named("x"),
            TypeName::Integer.tensor().named("axes"),
        ],
        &[("output", TypeName::Scalar.tensor())],
        log_softmax,
    );
}

pub fn log_softmax(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<Value> {
    let x = invocation.named_arg_as(builder, "x")?;
    let axes: TVec<usize> = invocation.named_arg_as(builder, "axes")?;

    let input_fact = builder.model.outlet_fact(x)?.clone();
    let quant_output_dt = if input_fact.datum_type.is_float() {
        None
    } else {
        invocation.dt_from_quant_file.first().cloned().flatten()
    };

    builder.wire(Softmax { axes, quant_output_dt, kind: SoftmaxKind::LogSoftmax }, &[x])
}
