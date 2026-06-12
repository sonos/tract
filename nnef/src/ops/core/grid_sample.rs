use crate::internal::*;
use tract_core::ops::nn::grid_sample::{GridSample, InterpolationMode, PaddingMode};

pub fn register(registry: &mut Registry) {
    registry.register_primitive(
        "tract_core_grid_sample",
        &parameters(),
        &[("output", TypeName::Scalar.tensor())],
        load,
    );
    registry.register_dumper(dump);
}

fn parameters() -> Vec<Parameter> {
    vec![
        TypeName::Scalar.tensor().named("input"),
        TypeName::Scalar.tensor().named("grid"),
        TypeName::String.named("mode").default("bilinear"),
        TypeName::String.named("padding_mode").default("zeros"),
        TypeName::Logical.named("align_corners").default(false),
    ]
}

fn dump(ast: &mut IntoAst, node: &TypedNode, op: &GridSample) -> TractResult<Option<Arc<RValue>>> {
    let input = ast.mapping[&node.inputs[0]].clone();
    let grid = ast.mapping[&node.inputs[1]].clone();
    Ok(Some(invocation(
        "tract_core_grid_sample",
        &[input, grid],
        &[
            ("mode", string(op.mode.as_str())),
            ("padding_mode", string(op.padding_mode.as_str())),
            ("align_corners", logical(op.align_corners)),
        ],
    )))
}

fn load(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let input = invocation.named_arg_as(builder, "input")?;
    let grid = invocation.named_arg_as(builder, "grid")?;
    let mode: String = invocation.named_arg_as(builder, "mode")?;
    let padding_mode: String = invocation.named_arg_as(builder, "padding_mode")?;
    let align_corners: bool = invocation.named_arg_as(builder, "align_corners")?;
    let op = GridSample {
        mode: InterpolationMode::parse(&mode)?,
        padding_mode: PaddingMode::parse(&padding_mode)?,
        align_corners,
    };
    builder.wire(op, &[input, grid])
}
