use crate::encoder::EncoderExt;
use crate::{LibraryName, MetalStream};
use anyhow::ensure;
use metal::{MTLSize, NSUInteger};
use tract_core::internal::*;
use tract_core::ops::element_wise::ElementWiseMiniOp;
use tract_gpu::tensor::DeviceTensor;

const ALL_OP_NAMES: &[&str] = &[
    "abs",
    "exp",
    "ln",
    "sigmoid",
    "square",
    "sqrt",
    "rsqrt",
    "recip",
    "ceil",
    "floor",
    "round",
    "roundhalftoeven",
    "cos",
    "acos",
    "acosh",
    "cosh",
    "sin",
    "asin",
    "asinh",
    "sinh",
    "tan",
    "atan",
    "atanh",
    "tanh",
    "erf",
    "neg",
    "sign",
    "hardswish",
    "silu",
];

pub fn all_functions() -> Vec<String> {
    ALL_OP_NAMES
        .iter()
        .flat_map(|kname| {
            DeviceTensor::SUPPORTED_DT.into_iter().flat_map(move |dt| {
                let tname = DeviceTensor::tname(dt).ok()?;
                Some(format!("element_wise_ops::{kname}_out_of_place_{tname}"))
            })
        })
        .collect()
}

pub fn is_supported(mini_op: &dyn ElementWiseMiniOp, dt: DatumType) -> bool {
    ALL_OP_NAMES.contains(&mini_op.name().to_lowercase().as_str())
        && matches!(dt, DatumType::F32 | DatumType::F16)
}

pub fn dispatch_eval(
    stream: &MetalStream,
    mini_op: &dyn ElementWiseMiniOp,
    input: &DeviceTensor,
    output: &DeviceTensor,
) -> TractResult<()> {
    stream.retain_tensor(input);
    stream.retain_tensor(output);

    ensure!(output.shape() == input.shape() && output.datum_type() == input.datum_type());

    let op_name = mini_op.name().to_lowercase();
    let tname = DeviceTensor::tname(input.datum_type())?;
    let kernel_name = format!("element_wise_ops::{op_name}_out_of_place_{tname}");

    let pipeline = stream.load_pipeline(LibraryName::ElementWiseOps, &kernel_name)?;
    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_metal_tensor(0, input, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(1, output, metal::MTLResourceUsage::Write);

        let grid_size = MTLSize { width: output.len() as NSUInteger, height: 1, depth: 1 };
        let group_size = MTLSize { width: 1, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid_size, group_size);
    });
    Ok(())
}

pub fn metal_element_wise_dispatch(
    mini_op: &dyn ElementWiseMiniOp,
    input: &DeviceTensor,
    output: &DeviceTensor,
) -> TractResult<()> {
    crate::with_metal_stream(|stream| dispatch_eval(stream, mini_op, input, output))
}

pub fn metal_element_wise_op(
    mini_op: Box<dyn ElementWiseMiniOp>,
) -> tract_gpu::ops::element_wise::GpuElementWise {
    tract_gpu::ops::element_wise::GpuElementWise::new(mini_op, "Metal", metal_element_wise_dispatch)
}

// Generic element-wise fallback — checked after LeakyRelu, GeluApproximate.
crate::register_metal_op!(tract_core::ops::element_wise::ElementWiseOp, |source, node, op| {
    rule_if!(is_supported(&*op.0, source.node_input_facts(node.id)?[0].datum_type));
    Ok(Some(Box::new(metal_element_wise_op(op.0.clone()))))
});
