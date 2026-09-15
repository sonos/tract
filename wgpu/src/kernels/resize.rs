use anyhow::ensure;
use tract_core::internal::*;
use tract_core::ops::nn::resize::Resize as CoreResize;
use tract_gpu::ops::resize::GpuResize;
use tract_gpu::tensor::DeviceTensor;

use crate::kernels::shaders::{
    EntryPoint, LayoutKind, ModuleKey, ModuleKind, PipelineKey, RESIZE_2D_WG, ShaderDtype,
    keys_for, pack_u32s, resize_2d_bilinear_module, resize_2d_module,
};
use crate::ops::resize2d::WgpuResize2d;
use crate::utils::{element_offset, get_wgpu_buffer};
use crate::with_wgpu_queue;

pub fn all_pipeline_keys(shader_f16: bool) -> Vec<PipelineKey> {
    keys_for(ModuleKind::Resize, &["resize_axis"], shader_f16)
}

/// Both trailing axes in one launch; `rows` and `cols` are (indices, weights,
/// window) plans as [`wgpu_resize_axis_dispatch`] takes them.
pub fn wgpu_resize_2d_dispatch(
    input: &DeviceTensor,
    rows: (&DeviceTensor, &DeviceTensor, usize),
    cols: (&DeviceTensor, &DeviceTensor, usize),
    output: &DeviceTensor,
) -> TractResult<()> {
    with_wgpu_queue(|q| {
        for t in [input, rows.0, rows.1, cols.0, cols.1, output] {
            q.retain_tensor(t);
        }
        let rank = input.rank();
        ensure!(rank >= 2 && output.rank() == rank);
        ensure!(
            rows.0.datum_type() == i32::datum_type() && cols.0.datum_type() == i32::datum_type()
        );
        ensure!(
            rows.1.datum_type() == f32::datum_type() && cols.1.datum_type() == f32::datum_type()
        );
        let dt = ShaderDtype::from_datum(input.datum_type())?;
        let layout = LayoutKind::Chain(6);
        let outer: usize = input.shape()[..rank - 2].iter().product::<usize>().max(1);
        let (h_in, w_in) = (input.shape()[rank - 2], input.shape()[rank - 1]);
        let (h_out, w_out) = (output.shape()[rank - 2], output.shape()[rank - 1]);
        let [wx, wy, wz] = RESIZE_2D_WG.map(|d| d as usize);
        let grid = [w_out.div_ceil(wx), h_out.div_ceil(wy), outer.div_ceil(wz)];
        let bilinear = rows.2 == 2 && cols.2 == 2 && grid.iter().all(|&g| g <= 65535);
        let pipeline = if bilinear {
            q.context().chain_pipeline(
                &format!("resize_2d_bilinear_{}", dt.suffix()),
                layout,
                EntryPoint::typed("resize_2d_bilinear", dt),
                || resize_2d_bilinear_module(dt),
            )?
        } else {
            q.context().chain_pipeline(
                &format!("resize_2d_{}", dt.suffix()),
                layout,
                EntryPoint::typed("resize_2d", dt),
                || resize_2d_module(dt),
            )?
        };
        let bg = q.context().bind_group(
            layout,
            &[
                get_wgpu_buffer(input),
                get_wgpu_buffer(rows.0),
                get_wgpu_buffer(rows.1),
                get_wgpu_buffer(cols.0),
                get_wgpu_buffer(cols.1),
                get_wgpu_buffer(output),
            ],
            q.uniform(),
        )?;
        let params = pack_u32s(&[
            element_offset(input, 0) as u32,
            element_offset(rows.0, 0) as u32,
            element_offset(rows.1, 0) as u32,
            element_offset(cols.0, 0) as u32,
            element_offset(cols.1, 0) as u32,
            element_offset(output, 0) as u32,
            outer as u32,
            h_in as u32,
            w_in as u32,
            h_out as u32,
            w_out as u32,
            rows.2 as u32,
            cols.2 as u32,
            0,
            0,
            0,
        ]);
        let dyn_off = q.alloc_uniform(&params)?;
        if bilinear {
            let grid = grid.map(|g| g as u32);
            return q.dispatch_grid("resize_2d_bilinear", &pipeline, &bg, dyn_off, grid);
        }
        q.dispatch("resize_2d", &pipeline, &bg, dyn_off, (outer * h_out * w_out) as u64)
    })
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
        Some(scales) => scales.cast_to::<f32>()?.try_as_plain_ram()?.as_slice::<f32>()?.into(),
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

/// The plans of a `GpuResize` over exactly the two trailing axes, which the
/// fused kernel serves.
pub struct Resize2dPlan {
    pub op: WgpuResize2d,
    /// row indices, row weights, column indices, column weights
    pub plans: [Arc<Tensor>; 4],
}

pub fn wgpu_resize_2d(source: &TypedModel, node: &TypedNode) -> TractResult<Option<Resize2dPlan>> {
    let Some(op) = node.op_as::<CoreResize>() else { return Ok(None) };
    let facts = source.node_input_facts(node.id)?;
    rule_if!(facts[0].is_plain());
    rule_if!(matches!(facts[0].datum_type, DatumType::F32 | DatumType::F16));
    let Some(baked) = baked_plans(source, node, op)? else { return Ok(None) };
    let rank = facts[0].rank();
    rule_if!(rank >= 2 && baked.axes.as_slice() == [rank - 2, rank - 1]);
    let [(ih, wh), (iw, ww)] = [&baked.plans[0], &baked.plans[1]];
    Ok(Some(Resize2dPlan {
        op: WgpuResize2d {
            window_h: baked.windows[0],
            window_w: baked.windows[1],
            output_shape: baked.output_shape.clone(),
        },
        plans: [ih.clone(), wh.clone(), iw.clone(), ww.clone()],
    }))
}

pub fn wgpu_resize(source: &TypedModel, node: &TypedNode) -> TractResult<Option<Box<dyn TypedOp>>> {
    let Some(op) = node.op_as::<CoreResize>() else { return Ok(None) };
    let facts = source.node_input_facts(node.id)?;
    rule_if!(facts[0].is_plain());
    rule_if!(matches!(facts[0].datum_type, DatumType::F32 | DatumType::F16));
    Ok(baked_plans(source, node, op)?.map(|op| Box::new(op) as Box<dyn TypedOp>))
}
