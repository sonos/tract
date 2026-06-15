//! Backward-compatible deserialization alias. GridSample now lives in
//! tract-core and serializes as `tract_core_grid_sample`; older NNEF dumps
//! named it `tract_onnx_grid_sample`, so keep a load-only primitive for them.
use tract_nnef::internal::*;
use tract_nnef::tract_core::ops::nn::grid_sample::{GridSample, InterpolationMode, PaddingMode};

pub fn register(registry: &mut Registry) {
    registry.register_primitive(
        "tract_onnx_grid_sample",
        &[
            TypeName::Scalar.tensor().named("input"),
            TypeName::Scalar.tensor().named("grid"),
            TypeName::String.named("mode").default("bilinear"),
            TypeName::String.named("padding_mode").default("zeros"),
            TypeName::Logical.named("align_corners").default(false),
        ],
        &[("output", TypeName::Scalar.tensor())],
        load,
    );
}

fn load(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let input = invocation.named_arg_as(builder, "input")?;
    let grid = invocation.named_arg_as(builder, "grid")?;
    let mode: String = invocation.named_arg_as(builder, "mode")?;
    let padding_mode: String = invocation.named_arg_as(builder, "padding_mode")?;
    let align_corners: bool = invocation.named_arg_as(builder, "align_corners")?;
    builder.wire(
        GridSample {
            mode: InterpolationMode::parse(&mode)?,
            padding_mode: PaddingMode::parse(&padding_mode)?,
            align_corners,
        },
        &[input, grid],
    )
}
