use crate::internal::*;
use tract_core::ops::lstm_seq::LstmSeq;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(ser_lstm_seq);
    registry.register_primitive(
        "tract_core_lstm_seq",
        &[
            TypeName::Scalar.tensor().named("x"),
            // Absent once `projected`: x already carries the input-side product.
            TypeName::Scalar.tensor().named("w").default(0),
            TypeName::Scalar.tensor().named("r"),
            TypeName::Scalar.tensor().named("b").default(0),
            TypeName::Scalar.tensor().named("initial_h"),
            TypeName::Scalar.tensor().named("initial_c"),
            TypeName::Integer.named("hidden"),
            TypeName::Integer.named("chunk"),
            // Explicit: an absent optional tensor resolves to its default, so `b`
            // alone cannot tell a bias-less LSTM from a zero-biased one.
            TypeName::Integer.spec().named("has_bias").default(0),
            TypeName::Integer.spec().named("reset_every_turn").default(0),
            TypeName::Integer.spec().named("emit_y").default(1),
            TypeName::Integer.spec().named("projected").default(0),
        ],
        &[
            ("y", TypeName::Scalar.tensor()),
            ("y_h", TypeName::Scalar.tensor()),
            ("y_c", TypeName::Scalar.tensor()),
        ],
        de_lstm_seq,
    );
}

fn de_lstm_seq(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let x = invocation.named_arg_as(builder, "x")?;
    let projected: bool = invocation.named_arg_as(builder, "projected")?;
    let w: Option<OutletId> =
        if projected { None } else { Some(invocation.named_arg_as(builder, "w")?) };
    let r = invocation.named_arg_as(builder, "r")?;
    let has_bias: bool = invocation.named_arg_as(builder, "has_bias")?;
    let b: Option<OutletId> =
        if has_bias { Some(invocation.named_arg_as(builder, "b")?) } else { None };
    let initial_h = invocation.named_arg_as(builder, "initial_h")?;
    let initial_c = invocation.named_arg_as(builder, "initial_c")?;
    let hidden: usize = invocation.named_arg_as(builder, "hidden")?;
    let chunk: i64 = invocation.named_arg_as(builder, "chunk")?;
    let reset_every_turn = invocation.named_arg_as(builder, "reset_every_turn")?;
    let emit_y = invocation.named_arg_as(builder, "emit_y")?;

    let mut inputs: TVec<OutletId> = tvec!(x);
    inputs.extend(w);
    inputs.push(r);
    inputs.extend(b);
    inputs.push(initial_h);
    inputs.push(initial_c);
    builder.wire(
        LstmSeq {
            hidden,
            has_bias,
            chunk: chunk as isize,
            reset_every_turn,
            emit_y,
            projected,
            packed_r: None,
        },
        &inputs,
    )
}

fn ser_lstm_seq(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &LstmSeq,
) -> TractResult<Option<Arc<RValue>>> {
    // b is optional and sits before the states, so every wire is named.
    let wire = |ix: usize| (*ast.mapping[&node.inputs[ix]]).clone();
    let mut named: Vec<(&str, RValue)> = vec![("x", wire(0))];
    let r_slot = if op.projected { 1 } else { 2 };
    if !op.projected {
        named.push(("w", wire(1)));
    }
    named.push(("r", wire(r_slot)));
    if op.has_bias {
        named.push(("b", wire(r_slot + 1)));
    }
    named.push(("initial_h", wire(node.inputs.len() - 2)));
    named.push(("initial_c", wire(node.inputs.len() - 1)));
    named.extend([
        ("hidden", numeric(op.hidden)),
        ("chunk", numeric(op.chunk)),
        ("has_bias", numeric(op.has_bias as usize)),
        ("reset_every_turn", numeric(op.reset_every_turn as usize)),
        ("emit_y", numeric(op.emit_y as usize)),
        ("projected", numeric(op.projected as usize)),
    ]);
    Ok(Some(invocation("tract_core_lstm_seq", &[], &named)))
}
