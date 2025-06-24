use tract_core::ops::nn::{Softmax, SoftmaxExp, SoftmaxKind};

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

    let exp: Option<String> = invocation.get_named_arg_as(builder, "exp")?;
    let exp = match exp.as_deref() {
        Some("fast_compact") => SoftmaxExp::FastCompact,
        _ => SoftmaxExp::Libc,
    };

    builder.wire(Softmax { axes, quant_output_dt, kind: SoftmaxKind::Softmax(exp) }, &[x])
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
    let mut args = vec![("axes", ints(&op.axes))];
    let op_name = match op.kind {
        SoftmaxKind::Softmax(exp) => {
            if exp == SoftmaxExp::FastCompact {
                args.push(("exp", string("fast_compact")))
            } else {
                return Ok(None);
            };
            "tract_core_softmax"
        }
        SoftmaxKind::LogSoftmax => "tract_core_log_softmax",
    };
    Ok(Some(invocation(op_name, &[wire], &args)))
}
