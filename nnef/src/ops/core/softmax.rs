use tract_core::ops::nn::{Softmax, SoftmaxKind};

use crate::{internal::*, ser::ints};

pub fn register(registry: &mut Registry) {
    registry.register_dumper(ser_softmax);
    registry.register_primitive(
        "tract_core_softmax",
        &[
            TypeName::Scalar.tensor().named("x"),
            TypeName::Integer.tensor().named("axes"),
            TypeName::String.named("exp"),
        ],
        &[("output", TypeName::Scalar.tensor())],
        deser_softmax,
    );
    registry.register_primitive(
        "tract_core_log_softmax",
        &[TypeName::Scalar.tensor().named("x"), TypeName::Integer.tensor().named("axes")],
        &[("output", TypeName::Scalar.tensor())],
        deser_log_softmax,
    );
}

pub fn deser_softmax(
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

    // `exp` is a legacy attribute (the old fast-compact selector). Accept it for
    // read-compat with graphs that still carry it, but ignore its value: softmax
    // always uses the accurate exp now.
    let _legacy_exp: Option<String> = invocation.get_named_arg_as(builder, "exp")?;

    builder.wire(Softmax { axes, quant_output_dt, kind: SoftmaxKind::Softmax }, &[x])
}

pub fn deser_log_softmax(
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

fn ser_softmax(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &Softmax,
) -> TractResult<Option<Arc<RValue>>> {
    let wire = ast.mapping[&node.inputs[0]].clone();
    // Plain softmax serializes through the standard NNEF path (returns None);
    // only log-softmax needs the tract_core extension.
    let op_name = match op.kind {
        SoftmaxKind::Softmax => return Ok(None),
        SoftmaxKind::LogSoftmax => "tract_core_log_softmax",
    };
    let args = vec![("axes", ints(&op.axes))];
    Ok(Some(invocation(op_name, &[wire], &args)))
}
