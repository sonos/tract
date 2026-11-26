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

#[cfg(test)]
mod tests {
    use crate::CudaTransform;
    use tract_core::internal::*;
    use tract_core::ops::cnn::{MaxPool, PaddingSpec, PoolSpec, SumPool};
    use tract_core::ops::nn::DataFormat;
    use tract_core::transform::ModelTransform;
    use tract_gpu::ops::pool::GpuPool;

    /// Max and average pools in every data format, with and without a batch axis, over one and
    /// two spatial axes: each must be translated to the device and match the CPU.
    #[test]
    fn pools_in_every_layout_run_on_device() -> TractResult<()> {
        let c = 5;
        for format in [DataFormat::NCHW, DataFormat::NHWC, DataFormat::CHW, DataFormat::HWC] {
            for hw in [vec![9usize, 11], vec![13]] {
                let rank = hw.len();
                let spec = PoolSpec::new(
                    format,
                    tvec![3; rank],
                    PaddingSpec::SameUpper,
                    None,
                    Some(tvec![2; rank]),
                    c,
                    c,
                );
                let shape = format.from_n_c_hw(2, c, &*hw)?.shape;
                for max in [true, false] {
                    let op: Box<dyn TypedOp> = if max {
                        Box::new(MaxPool::new(spec.clone(), None))
                    } else {
                        Box::new(SumPool::new(spec.clone(), false, true))
                    };
                    let case = format!("{format:?} {hw:?} max={max}");
                    let len: usize = shape.iter().product();
                    let values: Vec<f32> =
                        (0..len).map(|i| ((i * 37) % 101) as f32 - 50.0).collect();
                    let input = Tensor::from_shape(&shape, &values)?;
                    let mut model = TypedModel::default();
                    let i = model.add_source("i", f32::fact(&*shape))?;
                    let out = model.wire_node("pool", op, &[i])?;
                    model.select_output_outlets(&out)?;
                    let cpu =
                        model.clone().into_runnable()?.run(tvec![input.clone().into_tvalue()])?;
                    let cuda = CudaTransform.transform_into(model)?;
                    ensure!(
                        cuda.nodes().iter().any(|n| n.op_is::<GpuPool>()),
                        "{case}: the pool did not reach the device"
                    );
                    let got = cuda.into_runnable()?.run(tvec![input.into_tvalue()])?;
                    cpu[0]
                        .clone()
                        .into_tensor()
                        .close_enough(&got[0].clone().into_tensor(), Approximation::Approximate)
                        .with_context(|| case.clone())?;
                }
            }
        }
        Ok(())
    }
}
