use tract_core::ops::nn::{Softmax, SoftmaxExp, SoftmaxKind};

use crate::internal::*;
use crate::ser::*;

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

fn ser_softmax(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &Softmax,
) -> TractResult<Option<Arc<RValue>>> {
    match op.kind {
        SoftmaxKind::Softmax(exp) if exp == SoftmaxExp::default() => Ok(None),
        SoftmaxKind::Softmax(exp) => {
            let exp_str = match exp {
                SoftmaxExp::FastCompact => "fast_compact",
                SoftmaxExp::Libc => "libc",
            };
            Ok(Some(invocation(
                "tract_core_softmax",
                &[ast.mapping[&node.inputs[0]].clone()],
                &[("axes", ints(&op.axes)), ("exp", string(exp_str))],
            )))
        }
        SoftmaxKind::LogSoftmax => {
            let litteral_axes: Vec<_> = op.axes.iter().map(|&it| (it as i64).into()).collect();
            Ok(Some(invocation(
                "tract_core_log_softmax",
                &[ast.mapping[&node.inputs[0]].clone()],
                &[("axes", RValue::Literal(crate::ast::Literal::Array(litteral_axes)))],
            )))
        }
    }
}

fn deser_softmax(
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

fn deser_log_softmax(
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
