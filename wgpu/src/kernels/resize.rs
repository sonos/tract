use anyhow::ensure;
use tract_core::internal::*;
use tract_core::ops::nn::resize::Resize as CoreResize;
use tract_gpu::ops::resize::GpuResize;
use tract_gpu::tensor::DeviceTensor;

use crate::kernels::shaders::{
    EntryPoint, LayoutKind, ModuleKey, ModuleKind, PipelineKey, ShaderDtype, keys_for, pack_u32s,
};
use crate::utils::{element_offset, get_wgpu_buffer};
use crate::with_wgpu_queue;

pub fn all_pipeline_keys(shader_f16: bool) -> Vec<PipelineKey> {
    keys_for(ModuleKind::Resize, &["resize_axis"], shader_f16)
}

pub fn wgpu_resize_axis_dispatch(
    input: &DeviceTensor,
    axis: usize,
    indices: &DeviceTensor,
    weights: &DeviceTensor,
    window: usize,
    output: &DeviceTensor,
) -> TractResult<()> {
    with_wgpu_queue(|q| {
        q.retain_tensor(input);
        q.retain_tensor(indices);
        q.retain_tensor(weights);
        q.retain_tensor(output);
        ensure!(input.rank() > axis);
        ensure!(indices.datum_type() == i32::datum_type());
        ensure!(weights.datum_type() == f32::datum_type());
        let dt = ShaderDtype::from_datum(input.datum_type())?;
        let pipeline = q.context().pipeline(PipelineKey {
            module: ModuleKey { kind: ModuleKind::Resize, dtype: dt },
            entry: EntryPoint::typed("resize_axis", dt),
        })?;
        let len_in = input.shape()[axis];
        let len_out = output.shape()[axis];
        let inner: usize = input.shape()[axis + 1..].iter().product();
        let outer: usize = input.shape()[..axis].iter().product();
        let n = (outer.max(1) * len_out * inner.max(1)) as u64;
        let in_buf = get_wgpu_buffer(input);
        let idx_buf = get_wgpu_buffer(indices);
        let w_buf = get_wgpu_buffer(weights);
        let out_buf = get_wgpu_buffer(output);
        let bg = q.context().bind_group(
            LayoutKind::Resize,
            &[in_buf, idx_buf, w_buf, out_buf],
            q.uniform(),
        )?;
        let params = pack_u32s(&[
            element_offset(input, 0) as u32,
            element_offset(indices, 0) as u32,
            element_offset(weights, 0) as u32,
            element_offset(output, 0) as u32,
            outer.max(1) as u32,
            len_in as u32,
            len_out as u32,
            inner.max(1) as u32,
            window as u32,
            0,
            0,
            0,
        ]);
        let dyn_off = q.alloc_uniform(&params)?;
        q.dispatch("resize", &pipeline, &bg, dyn_off, n)
    })
}

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
    Ok(Some(GpuResize::new(axes, windows, plans, output_shape, "Wgpu", wgpu_resize_axis_dispatch)))
}

pub fn wgpu_resize(source: &TypedModel, node: &TypedNode) -> TractResult<Option<Box<dyn TypedOp>>> {
    let Some(op) = node.op_as::<CoreResize>() else { return Ok(None) };
    let facts = source.node_input_facts(node.id)?;
    rule_if!(facts[0].is_plain());
    rule_if!(matches!(facts[0].datum_type, DatumType::F32 | DatumType::F16));
    Ok(baked_plans(source, node, op)?.map(|op| Box::new(op) as Box<dyn TypedOp>))
}
