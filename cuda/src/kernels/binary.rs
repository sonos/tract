use cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg};
use std::fmt;
use tract_core::internal::*;
use tract_gpu::ops::binary::BinOp;
use tract_gpu::tensor::DeviceTensor;

use crate::context::{StreamExt, TractCudaStream, cuda_context};
use crate::kernels::launch_args::TractLaunchArgs;
use crate::kernels::utils::compute_broadcast_strides;
use crate::kernels::{LibraryName, MAX_THREADS, get_cuda_view};

static BINARY_MAX_RANK: usize = 5;

pub fn all_functions() -> Vec<String> {
    BinOp::ALL
        .into_iter()
        .flat_map(|op| DeviceTensor::SUPPORTED_DT.into_iter().map(move |dt| (op, dt)))
        .flat_map(|(op, dt)| ["large", "generic"].into_iter().map(move |variant| (op, dt, variant)))
        .flat_map(|(op, dt, variant)| kernel_name(op, dt, variant).into_iter())
        .collect()
}

pub fn kernel_name(op: BinOp, dt: DatumType, variant: &str) -> TractResult<String> {
    ensure!(op.is_supported_dt(dt), "Unsupported dt {:?} for Cuda binary ops: {op}", dt);

    let tname = DeviceTensor::tname(dt)?;

    let kname = match op {
        BinOp::Mul => "mul",
        BinOp::Add => "add",
        BinOp::Div => "div",
        BinOp::Sub => "sub",
        BinOp::Pow => "pow",
        BinOp::Min => "min",
        BinOp::Max => "max",
        BinOp::Greater => "greater",
        BinOp::GreaterEqual => "greater_equal",
        BinOp::Equals => "equals",
        BinOp::NotEquals => "not_equals",
        BinOp::Less => "less",
        BinOp::LessEqual => "less_equal",
        BinOp::And => "and",
        BinOp::Or => "or",
        BinOp::BitOr => "bitor",
        BinOp::BitAnd => "bitand",
        BinOp::BitXor => "bitxor",
    };

    Ok(format!("binary_{kname}_{variant}_{tname}"))
}

pub fn eval(
    stream: &TractCudaStream,
    op: BinOp,
    lhs: &DeviceTensor,
    rhs: &DeviceTensor,
) -> TractResult<DeviceTensor> {
    let out_shape = op.output_shape(lhs.shape(), rhs.shape())?;
    let out_dt = op.output_datum_type(lhs.datum_type(), rhs.datum_type())?;
    let output = unsafe { DeviceTensor::uninitialized_dt(out_dt, &out_shape)? };

    dispatch_eval(stream, op, lhs, rhs, &output)?;

    stream.synchronize()?;
    Ok(output)
}

pub fn dispatch_eval(
    stream: &TractCudaStream,
    op: BinOp,
    lhs: &DeviceTensor,
    rhs: &DeviceTensor,
    output: &DeviceTensor,
) -> TractResult<()> {
    let rank = lhs.rank();
    ensure!(rank == rhs.rank());
    ensure!(rank <= BINARY_MAX_RANK);

    let rank_offset = BINARY_MAX_RANK - rank;
    let mut lhs_shape = [1usize; BINARY_MAX_RANK];
    let mut rhs_shape = [1usize; BINARY_MAX_RANK];
    let mut out_shape = [1usize; BINARY_MAX_RANK];

    lhs_shape[rank_offset..].copy_from_slice(lhs.shape());
    rhs_shape[rank_offset..].copy_from_slice(rhs.shape());
    out_shape[rank_offset..].copy_from_slice(output.shape());

    let lhs_strides: TVec<usize> =
        compute_broadcast_strides(&lhs_shape, &Tensor::natural_strides(&lhs_shape))?;
    let rhs_strides: TVec<usize> =
        compute_broadcast_strides(&rhs_shape, &Tensor::natural_strides(&rhs_shape))?;
    let out_strides = Tensor::natural_strides(&out_shape);
    let n_el = out_shape.iter().product::<usize>();

    let use_large_kernel = n_el >= MAX_THREADS;
    let variant = if use_large_kernel { "large" } else { "generic" };

    let kname = kernel_name(op, lhs.datum_type(), variant)?;
    let func = cuda_context().load_pipeline(LibraryName::Binary, kname)?;

    let lhs_view = get_cuda_view(lhs);
    let rhs_view = get_cuda_view(rhs);
    let out_view = get_cuda_view(output);

    let mut launch_args = TractLaunchArgs::new(stream, &func);

    launch_args.push_view(&lhs_view);
    launch_args.push_slice_i32(&lhs_shape);
    launch_args.push_slice_i32(&lhs_strides);

    launch_args.push_view(&rhs_view);
    launch_args.push_slice_i32(&rhs_shape);
    launch_args.push_slice_i32(&rhs_strides);

    launch_args.push_view(&out_view);
    launch_args.push_slice_i32(&out_shape);
    launch_args.push_slice_i32(&out_strides);

    let cfg = if use_large_kernel {
        LaunchConfig {
            grid_dim: (n_el.div_ceil(MAX_THREADS) as u32, 1, 1),
            block_dim: (MAX_THREADS as u32, 1, 1),
            shared_mem_bytes: 0,
        }
    } else {
        LaunchConfig { grid_dim: (1, 1, 1), block_dim: (n_el as u32, 1, 1), shared_mem_bytes: 0 }
    };

    launch_args.launch(cfg)?;

    Ok(())
}

pub fn cuda_bin_op_dispatch(
    stream: &dyn tract_gpu::GpuStream,
    op: BinOp,
    lhs: &DeviceTensor,
    rhs: &DeviceTensor,
    output: &DeviceTensor,
) -> TractResult<()> {
    let stream = stream.cuda()?;
    dispatch_eval(stream, op, lhs, rhs, output)
}

#[cfg(test)]
mod tests {
    use tract_gpu::tensor::IntoDevice;

    use super::*;
    use crate::context::with_cuda_stream;
    use derive_new::new;
    use num_traits::AsPrimitive;
    use num_traits::Float;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use tract_core::internal::Tensor;

    fn test_case<F>(op: BinOp, shape: &[usize], offset: f32, scale: f32) -> TractResult<()>
    where
        F: Float + Datum,
        usize: AsPrimitive<f32>,
        f32: AsPrimitive<F>,
    {
        with_cuda_stream(|stream| {
            let stream = stream.cuda()?;
            let len = shape.iter().product::<usize>();

            let a = Tensor::from_shape(
                shape,
                &(0..len)
                    .map(|f| -> F {
                        let v: f32 = f.as_();
                        (v * scale + offset).as_()
                    })
                    .collect::<Vec<_>>(),
            )?
            .into_device()?;

            let b = Tensor::from_shape(
                shape,
                &(0..len)
                    .map(|f| -> F {
                        let v: f32 = f.as_();
                        (v * scale + offset + 1.0).as_()
                    })
                    .collect::<Vec<_>>(),
            )?
            .into_device()?;

            let cuda_output = eval(stream, op, &a, &b)?;

            // Just check it doesn't crash and produces the right shape
            let out = cuda_output.to_host()?.into_tensor();
            assert_eq!(out.shape(), shape);
            Ok(())
        })
    }

    #[test]
    fn test_binary_add() -> TractResult<()> {
        test_case::<f32>(BinOp::Add, &[4, 4], 0.0, 1.0)?;
        test_case::<f16>(BinOp::Add, &[4, 4], 0.0, 1.0 / 100.0)?;
        Ok(())
    }

    #[test]
    fn test_binary_mul() -> TractResult<()> {
        test_case::<f32>(BinOp::Mul, &[4, 4], 0.0, 1.0)?;
        test_case::<f16>(BinOp::Mul, &[4, 4], 0.0, 1.0 / 100.0)?;
        Ok(())
    }

    #[test]
    fn test_binary_sub() -> TractResult<()> {
        test_case::<f32>(BinOp::Sub, &[4, 4], 0.0, 1.0)?;
        Ok(())
    }

    #[test]
    fn test_binary_min() -> TractResult<()> {
        test_case::<f32>(BinOp::Min, &[4, 4], 0.0, 1.0)?;
        Ok(())
    }

    #[test]
    fn test_binary_max() -> TractResult<()> {
        test_case::<f32>(BinOp::Max, &[4, 4], 0.0, 1.0)?;
        Ok(())
    }
}
