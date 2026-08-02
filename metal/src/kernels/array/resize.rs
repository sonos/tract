use crate::encoder::EncoderExt;
use crate::{LibraryName, MetalStream};
use anyhow::ensure;
use metal::MTLSize;
use std::fmt;
use tract_core::internal::*;
use tract_core::ops::nn::resize::Resize as CoreResize;
use tract_gpu::ops::resize::GpuResize;
use tract_gpu::tensor::DeviceTensor;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ResizeAxis;

impl fmt::Display for ResizeAxis {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl ResizeAxis {
    pub fn is_supported_dt(dt: DatumType) -> bool {
        matches!(dt, DatumType::F32 | DatumType::F16)
    }

    pub fn kernel_name(&self, dt: DatumType) -> TractResult<String> {
        ensure!(Self::is_supported_dt(dt), "Unsupported dt {:?} for metal resize op", dt);
        let tname = DeviceTensor::tname(dt)?;
        Ok(format!("array_ops::resize_axis_{tname}"))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_eval(
        &self,
        stream: &MetalStream,
        input: &DeviceTensor,
        axis: usize,
        indices: &DeviceTensor,
        weights: &DeviceTensor,
        window: usize,
        output: &DeviceTensor,
    ) -> TractResult<()> {
        stream.retain_tensor(input);
        stream.retain_tensor(indices);
        stream.retain_tensor(weights);
        stream.retain_tensor(output);

        ensure!(input.rank() > axis);
        ensure!(output.datum_type() == input.datum_type());
        ensure!(indices.datum_type() == i32::datum_type());
        ensure!(weights.datum_type() == f32::datum_type());

        let len_in = input.shape()[axis];
        let len_out = output.shape()[axis];
        let inner: usize = input.shape()[axis + 1..].iter().product();
        let outer: usize = input.shape()[..axis].iter().product();
        ensure!(indices.len() == len_out * window && weights.len() == len_out * window);

        let params: [i32; 5] =
            [outer as i32, len_in as i32, len_out as i32, inner as i32, window as i32];

        let pipeline =
            stream.load_pipeline(LibraryName::ArrayOps, &self.kernel_name(input.datum_type())?)?;
        let command_buffer = stream.command_buffer();
        command_buffer.encode(|encoder| {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_metal_tensor(0, input, metal::MTLResourceUsage::Read);
            encoder.set_metal_tensor(1, output, metal::MTLResourceUsage::Write);
            encoder.set_metal_tensor(2, indices, metal::MTLResourceUsage::Read);
            encoder.set_metal_tensor(3, weights, metal::MTLResourceUsage::Read);
            encoder.set_slice(4, &params);
            let grid_size = MTLSize { width: inner as _, height: len_out as _, depth: outer as _ };
            let group_size = MTLSize { width: 1, height: 1, depth: 1 };
            encoder.dispatch_thread_groups(grid_size, group_size);
        });
        Ok(())
    }
}

pub fn metal_resize_axis_dispatch(
    input: &DeviceTensor,
    axis: usize,
    indices: &DeviceTensor,
    weights: &DeviceTensor,
    window: usize,
    output: &DeviceTensor,
) -> TractResult<()> {
    crate::with_metal_stream(|stream| {
        ResizeAxis.dispatch_eval(stream, input, axis, indices, weights, window, output)
    })
}

/// Bakes the resampling plan for every non-identity axis. Needs concrete input
/// dims and a constant scales/sizes input, so a symbolically-shaped or
/// dynamically-scaled Resize stays on CPU.
fn baked_plans(
    source: &TypedModel,
    node: &TypedNode,
    op: &CoreResize,
) -> TractResult<Option<GpuResize>> {
    let facts = source.node_input_facts(node.id)?;
    let Some(input_shape) = facts[0].shape.as_concrete() else { return Ok(None) };
    let aux = |ix: Option<usize>| ix.and_then(|ix| facts.get(ix)?.konst.as_deref());
    let scales_konst = aux(op.optional_scales_input);
    let sizes_konst = aux(op.optional_sizes_input);
    if scales_konst.is_none() && sizes_konst.is_none() {
        return Ok(None);
    }
    let output_shape: TVec<usize> =
        op.compute_output_shape(input_shape, scales_konst, sizes_konst)?;

    let scales: TVec<f32> = match scales_konst.filter(|s| s.len() == input_shape.len()) {
        Some(scales) => scales.cast_to::<f32>()?.try_as_plain()?.as_slice::<f32>()?.into(),
        None => output_shape.iter().zip(input_shape).map(|(o, i)| *o as f32 / *i as f32).collect(),
    };

    let (mut axes, mut windows, mut plans) = (tvec!(), tvec!(), tvec!());
    for (axis, &scale) in scales.iter().enumerate() {
        let (len_in, len_out) = (input_shape[axis], output_shape[axis]);
        if len_in == len_out && scale == 1.0 {
            continue;
        }
        let plan = op.plan_axis(scale, len_in, len_out);
        let indices: Vec<i32> = plan.indices.iter().map(|&i| i as i32).collect();
        plans.push((
            tract_ndarray::arr1(&indices).into_arc_tensor(),
            tract_ndarray::arr1(&plan.weights).into_arc_tensor(),
        ));
        axes.push(axis);
        windows.push(plan.window);
    }
    if axes.is_empty() {
        return Ok(None);
    }
    Ok(Some(GpuResize::new(
        axes,
        windows,
        plans,
        output_shape,
        "Metal",
        metal_resize_axis_dispatch,
    )))
}

/// Translates a core Resize whose plan can be baked. Wired by `transform.rs`
/// rather than `register_metal_op!` because the resulting op drops the
/// scales/sizes input, whose `TDim` datum type has no device equivalent.
pub fn metal_resize(
    source: &TypedModel,
    node: &TypedNode,
) -> TractResult<Option<Box<dyn TypedOp>>> {
    let Some(op) = node.op_as::<CoreResize>() else { return Ok(None) };
    let facts = source.node_input_facts(node.id)?;
    rule_if!(facts[0].is_plain());
    rule_if!(ResizeAxis::is_supported_dt(facts[0].datum_type));
    Ok(baked_plans(source, node, op)?.map(|op| Box::new(op) as Box<dyn TypedOp>))
}
