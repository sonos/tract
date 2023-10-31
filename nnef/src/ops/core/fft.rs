use crate::internal::*;
use tract_core::ops::fft::{Fft, Stft};

pub fn register(registry: &mut Registry) {
    registry.register_dumper(ser_fft);
    registry.register_primitive(
        "tract_core_fft",
        &[
            TypeName::Scalar.tensor().named("input"),
            TypeName::Integer.named("axis"),
            TypeName::Logical.named("inverse"),
        ],
        &[("output", TypeName::Scalar.tensor())],
        de_fft,
    );
    registry.register_dumper(ser_stft);
    registry.register_primitive(
        "tract_core_stft",
        &[
            TypeName::Scalar.tensor().named("input"),
            TypeName::Integer.named("axis"),
            TypeName::Integer.named("frame"),
            TypeName::Integer.named("stride"),
            TypeName::Scalar.tensor().named("window").default(false),
        ],
        &[("output", TypeName::Scalar.tensor())],
        de_stft,
    );
}

fn ser_fft(ast: &mut IntoAst, node: &TypedNode, op: &Fft) -> TractResult<Option<Arc<RValue>>> {
    let input = ast.mapping[&node.inputs[0]].clone();
    Ok(Some(invocation(
        "tract_core_fft",
        &[input],
        &[("axis", numeric(op.axis)), ("inverse", logical(op.inverse))],
    )))
}

fn de_fft(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let input = invocation.named_arg_as(builder, "input")?;
    let axis: usize = invocation.named_arg_as(builder, "axis")?;
    let inverse: bool = invocation.named_arg_as(builder, "inverse")?;
    let op = Fft { axis, inverse };
    builder.wire(op, &[input])
}

fn ser_stft(ast: &mut IntoAst, node: &TypedNode, op: &Stft) -> TractResult<Option<Arc<RValue>>> {
    let input = ast.mapping[&node.inputs[0]].clone();
    let mut named: TVec<(_, RValue)> = tvec![
        ("axis", numeric(op.axis)),
        ("frame", numeric(op.frame)),
        ("stride", numeric(op.stride)),
    ];
    if let Some(w) = &op.window {
        let w = ast.konst(format!("{}_window", node.name), w)?;
        named.push(("window", (*w).clone()));
    }
    Ok(Some(invocation("tract_core_stft", &[input], &named)))
}

fn de_stft(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let input = invocation.named_arg_as(builder, "input")?;
    let axis: usize = invocation.named_arg_as(builder, "axis")?;
    let frame: usize = invocation.named_arg_as(builder, "frame")?;
    let stride: usize = invocation.named_arg_as(builder, "stride")?;
    let window = invocation.optional_named_arg_as::<Arc<Tensor>>(builder, "window")?;
    let op = Stft { axis, frame, stride, window };
    builder.wire(op, &[input])
}
