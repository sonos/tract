use crate::context::cuda_context;
use crate::kernels::launch_args::TractLaunchArgs;
use crate::kernels::{LibraryName, get_cuda_view};
use cudarc::driver::{LaunchConfig, PushKernelArg};
use tract_core::internal::*;
use tract_gpu::ops::group_norm::GroupNormView;
use tract_gpu::tensor::DeviceTensor;

const BLOCK: u32 = 256;

pub fn is_supported_dt(dt: DatumType) -> bool {
    matches!(dt, DatumType::F32 | DatumType::F16)
}

pub fn kernel_names() -> Vec<String> {
    ["f32", "f16"]
        .into_iter()
        .flat_map(|t| [format!("group_norm_stats_{t}"), format!("group_norm_apply_{t}")])
        .collect()
}

fn launch_config(view: &GroupNormView) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((view.outer * view.groups) as _, view.splits as _, 1),
        block_dim: (BLOCK, 1, 1),
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
        let func =
            cuda_context().load_pipeline(LibraryName::NN, format!("group_norm_stats_{tname}"))?;
        let i_view = get_cuda_view(input);
        let p_view = get_cuda_view(partials);
        let mut args = TractLaunchArgs::new(stream, &func);
        args.push_view(&i_view);
        args.push_view(&p_view);
        args.push_slice_i32(&[view.pre, view.channels, view.post, view.groups, view.splits]);
        args.launch(launch_config(view))
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
        let tname = DeviceTensor::tname(input.datum_type())?;
        let func =
            cuda_context().load_pipeline(LibraryName::NN, format!("group_norm_apply_{tname}"))?;
        let i_view = get_cuda_view(input);
        let p_view = get_cuda_view(partials);
        let o_view = get_cuda_view(output);
        let mut args = TractLaunchArgs::new(stream, &func);
        args.push_view(&i_view);
        args.push_view(&p_view);
        args.push_view(&o_view);
        args.push_slice_i32(&[view.pre, view.channels, view.post, view.groups, view.splits]);
        args.push::<f32>(eps);
        args.launch(launch_config(view))
    })
}
