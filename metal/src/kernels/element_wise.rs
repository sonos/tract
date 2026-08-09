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
    "bitnot",
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

/// Limits of the fused elementwise chain kernel (keep in sync with
/// element_wise.metal).
pub const FUSED_EW_MAX_INPUTS: usize = 6;
pub const FUSED_EW_MAX_STEPS: usize = 24;
pub const FUSED_EW_MAX_STACK: usize = 8;
pub const FUSED_EW_MAX_RANK: usize = 4;

/// Opcodes of the fused elementwise chain interpreter (keep in sync with
/// element_wise.metal).
pub mod fused_ew_codes {
    pub const FLAG_ROUND_F16: u32 = 0x100;
    pub const SRC_SHIFT: u32 = 16;
    pub const PUSH_INPUT: u32 = 1;
    pub const PUSH_SCALAR: u32 = 2;
    pub const NEG: u32 = 16;
    pub const EXP: u32 = 17;
    pub const LN: u32 = 18;
    pub const SIGMOID: u32 = 19;
    pub const SILU: u32 = 20;
    pub const TANH: u32 = 21;
    pub const SQRT: u32 = 22;
    pub const RSQRT: u32 = 23;
    pub const RECIP: u32 = 24;
    pub const ABS: u32 = 25;
    pub const SQUARE: u32 = 26;
    pub const ID: u32 = 27;
    pub const ADD: u32 = 48;
    pub const SUB: u32 = 49;
    pub const MUL: u32 = 50;
    pub const DIV: u32 = 51;
    pub const MIN: u32 = 52;
    pub const MAX: u32 = 53;
    pub const POW: u32 = 54;
}

/// One step of the fused chain program as encoded for the GPU: `code` packs
/// `opcode | flags | input_index << 16`, `imm` is the f32 immediate of
/// PUSH_SCALAR steps.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FusedEwStepRaw {
    pub code: u32,
    pub imm: f32,
}

/// Runtime-only kernel parameters (the program itself is baked into the
/// pipeline as function constants). Layout mirrors FusedEwRtParams in
/// element_wise.metal.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct FusedEwRtParams {
    total: u32,
    out_shape: [u32; FUSED_EW_MAX_RANK],
    in_strides: [[u32; FUSED_EW_MAX_RANK]; FUSED_EW_MAX_INPUTS],
}

fn fused_ew_dt_code(dt: DatumType) -> TractResult<u32> {
    match dt {
        DatumType::F32 => Ok(0),
        DatumType::F16 => Ok(1),
        _ => bail!("fused elementwise chain only supports f16/f32, got {dt:?}"),
    }
}

/// Dispatches a fused chain. `steps` is the RPN program (codes already
/// carrying their round-to-f16 flags and source indexes); the program and the
/// input dtypes specialize the pipeline through function constants (compiled
/// once per distinct program, then served from the pipeline cache), while
/// shapes and broadcast strides stay runtime parameters.
pub fn dispatch_fused_elementwise_chain(
    stream: &MetalStream,
    steps: &[FusedEwStepRaw],
    inputs: &[&DeviceTensor],
    output: &DeviceTensor,
) -> TractResult<()> {
    use crate::func_constants::{ConstantValues, Value};

    ensure!(!inputs.is_empty() && inputs.len() <= FUSED_EW_MAX_INPUTS);
    ensure!(!steps.is_empty() && steps.len() <= FUSED_EW_MAX_STEPS);
    ensure!(output.rank() <= FUSED_EW_MAX_RANK);
    for input in inputs {
        stream.retain_tensor(input);
    }
    stream.retain_tensor(output);

    let total = output.len();
    if total == 0 {
        return Ok(());
    }

    let pad = |shape: &[usize]| -> [u32; FUSED_EW_MAX_RANK] {
        let mut out = [1u32; FUSED_EW_MAX_RANK];
        for (i, d) in shape.iter().rev().enumerate() {
            out[FUSED_EW_MAX_RANK - 1 - i] = *d as u32;
        }
        out
    };
    let out_shape = pad(output.shape());

    let mut params = FusedEwRtParams {
        total: total as u32,
        out_shape,
        in_strides: [[0; FUSED_EW_MAX_RANK]; FUSED_EW_MAX_INPUTS],
    };
    let mut in_f16_mask = 0usize;
    for (i, input) in inputs.iter().enumerate() {
        ensure!(input.rank() <= FUSED_EW_MAX_RANK);
        in_f16_mask |= (fused_ew_dt_code(input.datum_type())? as usize) << i;
        let in_shape = pad(input.shape());
        let mut stride = 1u32;
        for a in (0..FUSED_EW_MAX_RANK).rev() {
            ensure!(
                in_shape[a] == out_shape[a] || in_shape[a] == 1,
                "input {i} shape {:?} does not broadcast to output {:?}",
                input.shape(),
                output.shape()
            );
            params.in_strides[i][a] = if in_shape[a] == 1 { 0 } else { stride };
            stride *= in_shape[a];
        }
    }

    // 4-wide variant when every input reads the innermost axis contiguously
    // (stride 1) or broadcast (stride 0) and the output innermost dim is a
    // multiple of 4: amortizes the program per 4 elements.
    let vec4 = out_shape[FUSED_EW_MAX_RANK - 1] % 4 == 0
        && (0..inputs.len()).all(|i| params.in_strides[i][FUSED_EW_MAX_RANK - 1] <= 1);
    let (kernel_name, work_items) = if vec4 {
        ("fused_elementwise_chain_v4", total / 4)
    } else {
        ("fused_elementwise_chain", total)
    };

    let mut constants: Vec<(usize, Value)> = Vec::with_capacity(3 + 2 * FUSED_EW_MAX_STEPS);
    constants.push((0, Value::USize(steps.len())));
    constants.push((1, Value::Bool(output.datum_type() == DatumType::F16)));
    constants.push((2, Value::USize(in_f16_mask)));
    for s in 0..FUSED_EW_MAX_STEPS {
        let step = steps.get(s).copied().unwrap_or_default();
        constants.push((10 + s, Value::USize(step.code as usize)));
        constants.push((40 + s, Value::F32(step.imm)));
    }

    let pipeline = stream.load_pipeline_with_constants(
        LibraryName::ElementWiseOps,
        kernel_name,
        Some(ConstantValues::new(constants)),
    )?;
    let group_width =
        (pipeline.max_total_threads_per_threadgroup() as usize).min(256).min(work_items);
    let grid_width = work_items.div_ceil(group_width);
    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        for slot in 0..FUSED_EW_MAX_INPUTS {
            // Unused slots are bound to input 0: Metal requires every declared
            // buffer argument bound, and the program never reads past
            // inputs.len().
            let t = inputs.get(slot).unwrap_or(&inputs[0]);
            encoder.set_metal_tensor(slot as u64, t, metal::MTLResourceUsage::Read);
        }
        encoder.set_metal_tensor(6, output, metal::MTLResourceUsage::Write);
        encoder.set_bytes(
            7,
            std::mem::size_of::<FusedEwRtParams>() as u64,
            &params as *const FusedEwRtParams as *const std::ffi::c_void,
        );
        let grid_size = MTLSize { width: grid_width as NSUInteger, height: 1, depth: 1 };
        let group_size = MTLSize { width: group_width as NSUInteger, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid_size, group_size);
    });
    Ok(())
}

pub fn is_supported(mini_op: &dyn ElementWiseMiniOp, dt: DatumType) -> bool {
    let name = mini_op.name().to_lowercase();
    ALL_OP_NAMES.contains(&name.as_str())
        && if name == "bitnot" {
            dt.is_integer() || dt.is::<bool>()
        } else {
            matches!(dt, DatumType::F32 | DatumType::F16)
        }
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
    let total = output.len() as u64;
    if total == 0 {
        return Ok(());
    }
    let mut group_width = (pipeline.max_total_threads_per_threadgroup() as u64).min(256).min(total);
    while total % group_width != 0 {
        group_width -= 1;
    }
    let grid_width = total / group_width;
    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_metal_tensor(0, input, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(1, output, metal::MTLResourceUsage::Write);

        let grid_size = MTLSize { width: grid_width as NSUInteger, height: 1, depth: 1 };
        let group_size = MTLSize { width: group_width as NSUInteger, height: 1, depth: 1 };
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
