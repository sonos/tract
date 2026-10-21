use crate::context::cuda_context;
use crate::kernels::launch_args::TractLaunchArgs;
use crate::kernels::{LibraryName, get_cuda_view};
use cudarc::driver::LaunchConfig;
use tract_core::internal::*;
use tract_core::ops::cnn::{MaxPool, OptMaxPool, OptSumPool, PoolSpec, SumPool};
use tract_gpu::ops::pool::{GpuPool, PoolKind};
use tract_gpu::tensor::DeviceTensor;

pub fn is_supported_dt(dt: DatumType) -> bool {
    matches!(dt, DatumType::F32 | DatumType::F16)
}

pub fn kernel_name(kind: PoolKind, dt: DatumType) -> TractResult<String> {
    ensure!(is_supported_dt(dt), "Unsupported dt {dt:?} for cuda pool");
    let stem = match kind {
        PoolKind::Max => "max",
        PoolKind::Sum { .. } => "sum",
    };
    Ok(format!("{stem}_pool_2d_{}", DeviceTensor::tname(dt)?))
}

/// Pools `input` into `output` over a 1D or 2D window, in any data format.
pub fn cuda_pool_dispatch(
    pool_spec: &PoolSpec,
    kind: PoolKind,
    input: &DeviceTensor,
    output: &DeviceTensor,
) -> TractResult<()> {
    ensure!(output.datum_type() == input.datum_type());
    ensure!(input.len() <= i32::MAX as usize, "CUDA pool takes i32 strides");
    let in_shape = pool_spec.data_format.shape(input.shape())?;
    let out_shape = pool_spec.data_format.shape(output.shape())?;
    let rank = in_shape.hw_rank();
    ensure!(rank == 1 || rank == 2, "CUDA pool is 1D or 2D");
    let padding = pool_spec.computed_padding(in_shape.hw_dims());
    // Spatial values as (h, w); a 1D pool gets the unit h.
    let hw = |unit: usize, v: &[usize]| if rank == 2 { (v[0], v[1]) } else { (unit, v[0]) };
    let (ih, iw) = hw(1, in_shape.hw_dims());
    let (oh, ow) = hw(1, out_shape.hw_dims());
    let (kh, kw) = hw(1, &pool_spec.kernel_shape);
    let (stride_h, stride_w) = hw(1, &pool_spec.strides());
    let (dil_h, dil_w) = hw(1, &pool_spec.dilations());
    let (pad_h, pad_w) = hw(0, &padding.iter().map(|p| p.pad_before).collect::<TVec<_>>());
    let (pad_after_h, pad_after_w) =
        hw(0, &padding.iter().map(|p| p.pad_after).collect::<TVec<_>>());
    let (in_sh, in_sw) = hw(0, in_shape.hw_strides());
    let (out_sh, out_sw) = hw(0, out_shape.hw_strides());
    let (count_include_pad, normalize) = match kind {
        PoolKind::Max => (false, false),
        PoolKind::Sum { count_include_pad, normalize } => (count_include_pad, normalize),
    };
    let n = *in_shape.n().unwrap_or(&1);
    let c = *in_shape.c();
    crate::with_cuda_stream(|stream| {
        let func = cuda_context()
            .load_pipeline(LibraryName::NN, kernel_name(kind, input.datum_type())?)?;
        let input_view = get_cuda_view(input);
        let output_view = get_cuda_view(output);
        let mut launcher = TractLaunchArgs::new(stream, &func);
        launcher.push_view(&input_view);
        launcher.push_view(&output_view);
        launcher.push_slice_i32(&[
            n,
            c,
            ih,
            iw,
            oh,
            ow,
            kh,
            kw,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            pad_after_h,
            pad_after_w,
            dil_h,
            dil_w,
            *in_shape.n_stride().unwrap_or(&0),
            *in_shape.c_stride(),
            in_sh,
            in_sw,
            *out_shape.n_stride().unwrap_or(&0),
            *out_shape.c_stride(),
            out_sh,
            out_sw,
            pool_spec.data_format.c_is_last() as usize,
            count_include_pad as usize,
            normalize as usize,
        ]);
        launcher.launch(LaunchConfig::for_num_elems(output.len() as u32))
    })
}

fn try_make_cuda_pool(
    source: &TypedModel,
    node: &TypedNode,
) -> TractResult<Option<Box<dyn TypedOp>>> {
    let facts = source.node_input_facts(node.id)?;
    rule_if!(facts[0].is_plain() && is_supported_dt(facts[0].datum_type));
    rule_if_some!(op = GpuPool::from_core(node.op.as_ref(), "Cuda", cuda_pool_dispatch));
    rule_if!(matches!(op.pool_spec.kernel_shape.len(), 1 | 2));
    Ok(Some(Box::new(op)))
}

crate::register_cuda_op!(MaxPool, |source, node, _op| try_make_cuda_pool(source, node));
crate::register_cuda_op!(SumPool, |source, node, _op| try_make_cuda_pool(source, node));
crate::register_cuda_op!(OptMaxPool, |source, node, _op| try_make_cuda_pool(source, node));
crate::register_cuda_op!(OptSumPool, |source, node, _op| try_make_cuda_pool(source, node));
