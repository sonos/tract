use crate::internal::*;
use tract_core::ops::gru_cell::{GruCell, GruEpilogue};

pub fn register(registry: &mut Registry) {
    registry.register_dumper(ser_gru_epilogue);
    registry.register_dumper(ser_gru_cell);
    registry.register_primitive(
        "tract_core_gru_cell",
        &[
            TypeName::Scalar.tensor().named("xh"),
            TypeName::Scalar.tensor().named("r"),
            TypeName::Scalar.tensor().named("rb"),
            TypeName::Scalar.tensor().named("h_prev"),
            TypeName::Integer.named("hidden"),
        ],
        &[("output", TypeName::Scalar.tensor())],
        de_gru_cell,
    );
    registry.register_primitive(
        "tract_core_gru_epilogue",
        &[
            TypeName::Scalar.tensor().named("xh"),
            TypeName::Scalar.tensor().named("rh"),
            TypeName::Scalar.tensor().named("h_prev"),
            TypeName::Integer.named("hidden"),
        ],
        &[("output", TypeName::Scalar.tensor())],
        de_gru_epilogue,
    );
}

fn de_gru_epilogue(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<Value> {
    let xh = invocation.named_arg_as(builder, "xh")?;
    let rh = invocation.named_arg_as(builder, "rh")?;
    let h_prev = invocation.named_arg_as(builder, "h_prev")?;
    let hidden: usize = invocation.named_arg_as(builder, "hidden")?;
    builder.wire(GruEpilogue { hidden }, &[xh, rh, h_prev])
}

fn de_gru_cell(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let xh = invocation.named_arg_as(builder, "xh")?;
    let r = invocation.named_arg_as(builder, "r")?;
    let rb = invocation.named_arg_as(builder, "rb")?;
    let h_prev = invocation.named_arg_as(builder, "h_prev")?;
    let hidden: usize = invocation.named_arg_as(builder, "hidden")?;
    builder.wire(GruCell { hidden }, &[xh, r, rb, h_prev])
}

fn ser_gru_cell(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &GruCell,
) -> TractResult<Option<Arc<RValue>>> {
    let wires: TVec<Arc<RValue>> = node.inputs.iter().map(|i| ast.mapping[i].clone()).collect();
    Ok(Some(invocation("tract_core_gru_cell", &wires, &[("hidden", numeric(op.hidden))])))
}

fn ser_gru_epilogue(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &GruEpilogue,
) -> TractResult<Option<Arc<RValue>>> {
    let xh = ast.mapping[&node.inputs[0]].clone();
    let rh = ast.mapping[&node.inputs[1]].clone();
    let h_prev = ast.mapping[&node.inputs[2]].clone();
    Ok(Some(invocation(
        "tract_core_gru_epilogue",
        &[xh, rh, h_prev],
        &[("hidden", numeric(op.hidden))],
    )))
}
