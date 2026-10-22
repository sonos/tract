use crate::context::cuda_context;
use crate::kernels::launch_args::TractLaunchArgs;
use crate::kernels::{LibraryName, get_cuda_view};
use cudarc::driver::{LaunchConfig, PushKernelArg};
use tract_core::internal::*;
use tract_gpu::ops::group_norm::GroupNormView;
use tract_gpu::tensor::DeviceTensor;

const BLOCK: usize = 256;

pub fn is_supported_dt(dt: DatumType) -> bool {
    matches!(dt, DatumType::F32 | DatumType::F16)
}

pub fn kernel_names() -> Vec<String> {
    ["f32", "f16"]
        .into_iter()
        .flat_map(|t| {
            [
                format!("group_norm_stats_{t}"),
                format!("group_norm_stats_channel_last_{t}"),
                format!("group_norm_apply_{t}"),
            ]
        })
        .chain(["group_norm_finalize".to_string()])
        .collect()
}

fn block_config(grid: (usize, usize)) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (grid.0 as _, grid.1 as _, 1),
        block_dim: (BLOCK as _, 1, 1),
        shared_mem_bytes: 0,
    }
}

pub fn cuda_group_norm_stats_dispatch(
    input: &DeviceTensor,
    view: &GroupNormView,
    partials: &DeviceTensor,
) -> TractResult<()> {
    ensure!(is_supported_dt(input.datum_type()));
    crate::with_cuda_stream(|stream| {
        let tname = DeviceTensor::tname(input.datum_type())?;
        let i_view = get_cuda_view(input);
        let p_view = get_cuda_view(partials);
        if view.channel_last {
            let func = cuda_context()
                .load_pipeline(LibraryName::NN, format!("group_norm_stats_channel_last_{tname}"))?;
            let mut args = TractLaunchArgs::new(stream, &func);
            args.push_view(&i_view);
            args.push_view(&p_view);
            args.push_slice_i32(&[view.pre, view.channels, view.splits]);
            let grid = view.outer * view.channels.div_ceil(BLOCK);
            args.launch(block_config((grid, view.splits)))
        } else {
            let func = cuda_context()
                .load_pipeline(LibraryName::NN, format!("group_norm_stats_{tname}"))?;
            let mut args = TractLaunchArgs::new(stream, &func);
            args.push_view(&i_view);
            args.push_view(&p_view);
            args.push_slice_i32(&[view.pre, view.channels, view.post, view.groups, view.splits]);
            args.launch(block_config((view.outer * view.groups, view.splits)))
        }
    })
}

pub fn cuda_group_norm_apply_dispatch(
    input: &DeviceTensor,
    partials: &DeviceTensor,
    view: &GroupNormView,
    eps: f32,
    output: &DeviceTensor,
) -> TractResult<()> {
    ensure!(is_supported_dt(input.datum_type()));
    crate::with_cuda_stream(|stream| {
        let p_view = get_cuda_view(partials);
        let func = cuda_context().load_pipeline(LibraryName::NN, "group_norm_finalize".into())?;
        let mut args = TractLaunchArgs::new(stream, &func);
        args.push_view(&p_view);
        args.push_i32(view.entries);
        args.push::<f32>(eps);
        args.launch(block_config((view.outer * view.groups, 1)))?;

        let tname = DeviceTensor::tname(input.datum_type())?;
        let func =
            cuda_context().load_pipeline(LibraryName::NN, format!("group_norm_apply_{tname}"))?;
        let i_view = get_cuda_view(input);
        let o_view = get_cuda_view(output);
        let mut args = TractLaunchArgs::new(stream, &func);
        args.push_view(&i_view);
        args.push_view(&p_view);
        args.push_view(&o_view);
        args.push_slice_i32(&[
            input.len(),
            view.pre,
            view.channels,
            view.post,
            view.groups,
            view.entries,
        ]);
        args.launch(block_config((input.len().div_ceil(BLOCK), 1)))
    })
}
