use crate::internal::*;
use tract_core::ops::gru_seq::GruSeq;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(ser_gru_seq);
    registry.register_primitive(
        "tract_core_gru_seq",
        &[
            TypeName::Scalar.tensor().named("x"),
            TypeName::Scalar.tensor().named("w"),
            TypeName::Scalar.tensor().named("r"),
            TypeName::Scalar.tensor().named("b").default(0),
            TypeName::Scalar.tensor().named("initial_h"),
            TypeName::Integer.named("hidden"),
            TypeName::Integer.named("chunk"),
            // Explicit rather than inferred from `b` being present: the optional
            // tensor arg resolves to its declared default when absent, so its
            // presence cannot tell a bias-less GRU from a zero-biased one.
            TypeName::Integer.spec().named("has_bias").default(0),
            // The state contract travels with the model, as Scan's does.
            TypeName::Integer.spec().named("reset_every_turn").default(0),
        ],
        &[("y", TypeName::Scalar.tensor()), ("y_h", TypeName::Scalar.tensor())],
        de_gru_seq,
    );
}

fn de_gru_seq(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let x = invocation.named_arg_as(builder, "x")?;
    let w = invocation.named_arg_as(builder, "w")?;
    let r = invocation.named_arg_as(builder, "r")?;
    let has_bias: bool = invocation.named_arg_as(builder, "has_bias")?;
    let b: Option<OutletId> =
        if has_bias { Some(invocation.named_arg_as(builder, "b")?) } else { None };
    let initial_h = invocation.named_arg_as(builder, "initial_h")?;
    let hidden: usize = invocation.named_arg_as(builder, "hidden")?;
    let chunk: i64 = invocation.named_arg_as(builder, "chunk")?;
    let reset_every_turn = invocation.named_arg_as(builder, "reset_every_turn")?;

    let mut inputs: TVec<OutletId> = tvec!(x, w, r);
    inputs.extend(b);
    inputs.push(initial_h);
    builder.wire(
        GruSeq { hidden, has_bias, chunk: chunk as isize, reset_every_turn },
        &inputs,
    )
}

fn ser_gru_seq(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &GruSeq,
) -> TractResult<Option<Arc<RValue>>> {
    // b sits between r and initial_h and is optional, so every wire is named:
    // dumping them positionally would bind initial_h to b on a bias-less GRU.
    let wire = |ix: usize| (*ast.mapping[&node.inputs[ix]]).clone();
    let mut named: Vec<(&str, RValue)> =
        vec![("x", wire(0)), ("w", wire(1)), ("r", wire(2))];
    if op.has_bias {
        named.push(("b", wire(3)));
    }
    named.push(("initial_h", wire(node.inputs.len() - 1)));
    named.extend([
        ("hidden", numeric(op.hidden)),
        ("chunk", numeric(op.chunk)),
        ("has_bias", numeric(op.has_bias as usize)),
        ("reset_every_turn", numeric(op.reset_every_turn as usize)),
    ]);
    Ok(Some(invocation("tract_core_gru_seq", &[], &named)))
}
