use crate::internal::*;
use tract_core::ops::lstm_cell::LstmEpilogue;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(ser_lstm_epilogue);
    registry.register_primitive(
        "tract_core_lstm_epilogue",
        &[
            TypeName::Scalar.tensor().named("preact"),
            TypeName::Scalar.tensor().named("c_prev"),
            TypeName::Integer.named("hidden"),
        ],
        &[("ht", TypeName::Scalar.tensor()), ("ct", TypeName::Scalar.tensor())],
        de_lstm_epilogue,
    );
}

fn de_lstm_epilogue(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<Value> {
    let preact = invocation.named_arg_as(builder, "preact")?;
    let c_prev = invocation.named_arg_as(builder, "c_prev")?;
    let hidden: usize = invocation.named_arg_as(builder, "hidden")?;
    builder.wire(LstmEpilogue { hidden }, &[preact, c_prev])
}

fn ser_lstm_epilogue(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &LstmEpilogue,
) -> TractResult<Option<Arc<RValue>>> {
    let preact = ast.mapping[&node.inputs[0]].clone();
    let c_prev = ast.mapping[&node.inputs[1]].clone();
    Ok(Some(invocation(
        "tract_core_lstm_epilogue",
        &[preact, c_prev],
        &[("hidden", numeric(op.hidden))],
    )))
}
