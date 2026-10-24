use tract_core::internal::*;
use tract_gpu::ops::group_norm::GroupNormView;
use tract_gpu::tensor::DeviceTensor;

use crate::kernels::shaders::{
    EntryPoint, GROUP_NORM_WG, LayoutKind, ShaderDtype, group_norm_apply_module,
    group_norm_finalize_module, group_norm_stats_module, pack_u32s,
};
use crate::utils::{element_offset, get_wgpu_buffer};
use crate::with_wgpu_queue;

pub fn wgpu_group_norm_stats_dispatch(
    input: &DeviceTensor,
    view: &GroupNormView,
    partials: &DeviceTensor,
) -> TractResult<()> {
    with_wgpu_queue(|q| {
        q.retain_tensor(input);
        q.retain_tensor(partials);
        let dt = ShaderDtype::from_datum(input.datum_type())?;
        let (stem, groups) = if view.channel_last {
            let blocks = view.outer * view.channels.div_ceil(GROUP_NORM_WG as usize);
            (EntryPoint::typed("group_norm_stats_channel_last", dt), blocks)
        } else {
            (EntryPoint::typed("group_norm_stats", dt), view.outer * view.groups)
        };
        ensure!(groups <= 65535, "group norm over {groups} workgroups exceeds one grid dimension");
        let layout = LayoutKind::Unary;
        let pipeline = q.context().chain_pipeline(
            ("group_norm_stats", dt, view.channel_last),
            layout,
            stem,
            || group_norm_stats_module(dt),
        )?;
        let bg = q.context().bind_group(
            layout,
            &[get_wgpu_buffer(input), get_wgpu_buffer(partials)],
            q.uniform(),
        )?;
        let params = pack_u32s(&[
            element_offset(input, 0) as u32,
            element_offset(partials, 0) as u32,
            view.pre as u32,
            view.channels as u32,
            view.post as u32,
            view.groups as u32,
            view.splits as u32,
            0,
        ]);
        let dyn_off = q.alloc_uniform(&params)?;
        q.dispatch_grid(
            "group_norm_stats",
            &pipeline,
            &bg,
            dyn_off,
            [groups as u32, view.splits as u32, 1],
        )
    })
}

/// Merges the partials into a `(mean, inverse std)` pair per group, then
/// normalizes the input with them.
pub fn wgpu_group_norm_apply_dispatch(
    input: &DeviceTensor,
    partials: &DeviceTensor,
    view: &GroupNormView,
    eps: f32,
    output: &DeviceTensor,
) -> TractResult<()> {
    with_wgpu_queue(|q| {
        q.retain_tensor(input);
        q.retain_tensor(partials);
        q.retain_tensor(output);
        let dt = ShaderDtype::from_datum(input.datum_type())?;
        let groups = view.outer * view.groups;
        ensure!(groups <= 65535, "group norm over {groups} groups exceeds one grid dimension");
        let stats = DeviceTensor::uninitialized_dt(DatumType::F32, &[groups, 2])?;
        q.retain_tensor(&stats);

        let layout = LayoutKind::Unary;
        let pipeline = q.context().chain_pipeline(
            "group_norm_finalize",
            layout,
            EntryPoint::plain("group_norm_finalize"),
            group_norm_finalize_module,
        )?;
        let bg = q.context().bind_group(
            layout,
            &[get_wgpu_buffer(partials), get_wgpu_buffer(&stats)],
            q.uniform(),
        )?;
        let params = pack_u32s(&[
            element_offset(partials, 0) as u32,
            element_offset(&stats, 0) as u32,
            view.entries as u32,
            eps.to_bits(),
        ]);
        let dyn_off = q.alloc_uniform(&params)?;
        q.dispatch_grid("group_norm_finalize", &pipeline, &bg, dyn_off, [groups as u32, 1, 1])?;

        let layout = LayoutKind::Binary;
        let pipeline = q.context().chain_pipeline(
            ("group_norm_apply", dt),
            layout,
            EntryPoint::typed("group_norm_apply", dt),
            || group_norm_apply_module(dt),
        )?;
        let bg = q.context().bind_group(
            layout,
            &[get_wgpu_buffer(input), get_wgpu_buffer(&stats), get_wgpu_buffer(output)],
            q.uniform(),
        )?;
        let params = pack_u32s(&[
            element_offset(input, 0) as u32,
            element_offset(&stats, 0) as u32,
            element_offset(output, 0) as u32,
            input.len() as u32,
            view.pre as u32,
            view.channels as u32,
            view.post as u32,
            view.groups as u32,
        ]);
        let dyn_off = q.alloc_uniform(&params)?;
        q.dispatch("group_norm_apply", &pipeline, &bg, dyn_off, input.len() as u64)
    })
}
