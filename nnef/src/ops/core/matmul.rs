use crate::internal::*;
use tract_core::ops::einsum::EinSum;

use super::qmatmul::from_legacy_axes_spec;

pub fn register(registry: &mut Registry) {
    registry.register_primitive(
        "tract_core_matmul",
        &matmul_parameters(),
        &[("output", TypeName::Scalar.tensor())],
        matmul_load,
    );
}

fn matmul_parameters() -> Vec<Parameter> {
    vec![
        TypeName::Scalar.tensor().named("A"),
        TypeName::Scalar.tensor().named("B"),
        TypeName::Integer.array().named("axes"),
    ]
}

fn matmul_load(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let a: OutletId = invocation.named_arg_as(builder, "A")?;
    let b: OutletId = invocation.named_arg_as(builder, "B")?;
    let axes: TVec<usize> = invocation.named_arg_as(builder, "axes")?;
    let fact = builder.model.outlet_fact(a)?;
    let axes = from_legacy_axes_spec(&axes, fact.rank())?;
    builder.wire(EinSum::new(axes, fact.datum_type), &[a, b])
}
