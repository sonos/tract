use cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg};
use tract_core::internal::*;
use tract_core::ops::binary::BinMiniOp;
use tract_gpu::tensor::DeviceTensor;

use crate::context::{TractCudaStream, cuda_context};
use crate::kernels::launch_args::TractLaunchArgs;
use crate::kernels::{LibraryName, get_cuda_view};

static BINARY_MAX_RANK: usize = 5;

const ALL_OP_NAMES: &[&str] = &[
    "mul", "add", "div", "sub", "pow", "min", "max", "gt", "gte", "eq", "ne", "lt", "lte", "and",
    "or", "bitor", "bitand", "bitxor",
];

pub fn all_functions() -> Vec<String> {
    ALL_OP_NAMES
        .iter()
        .flat_map(|kname| {
            DeviceTensor::SUPPORTED_DT.into_iter().flat_map(move |dt| {
                let tname = DeviceTensor::tname(dt).ok()?;
                Some(
                    ["large", "generic"]
                        .into_iter()
                        .map(move |variant| format!("binary_{kname}_{variant}_{tname}")),
                )
            })
        })
        .flatten()
        .collect()
}

pub fn is_supported(mini_op: &dyn BinMiniOp, dt: DatumType) -> bool {
    ALL_OP_NAMES.contains(&mini_op.name().to_lowercase().as_str())
        && (dt.is_number() || dt.is::<bool>())
}

pub fn dispatch_eval(
    stream: &TractCudaStream,
    mini_op: &dyn BinMiniOp,
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
    let mut lhs_strides = [0isize; BINARY_MAX_RANK];
    let mut rhs_strides = [0isize; BINARY_MAX_RANK];
    let mut out_strides = [0isize; BINARY_MAX_RANK];

    let base_l_shape = lhs.shape();
    let base_r_shape = rhs.shape();
    let base_o_shape = output.shape();
    let base_l_strides = lhs.strides();
    let base_r_strides = rhs.strides();
    let base_o_strides = output.strides();
    for i in 0..rank {
        let dst = rank_offset + i;
        lhs_shape[dst] = base_l_shape[i];
        rhs_shape[dst] = base_r_shape[i];
        out_shape[dst] = base_o_shape[i];
        lhs_strides[dst] =
            if base_l_shape[i] == 1 && base_r_shape[i] != 1 { 0 } else { base_l_strides[i] };
        rhs_strides[dst] =
            if base_r_shape[i] == 1 && base_l_shape[i] != 1 { 0 } else { base_r_strides[i] };
        out_strides[dst] = base_o_strides[i];
    }

    let total_elems: usize = out_shape.iter().product();
    let block_dim = (128_u32, 1, 1);
    let (grid_dim, variant) = if out_shape[BINARY_MAX_RANK - 1] >= 256 && total_elems >= 4096 {
        (
            (
                out_shape[BINARY_MAX_RANK - 2] as u32,
                out_shape[BINARY_MAX_RANK - 3] as u32,
                out_shape[..BINARY_MAX_RANK - 3].iter().product::<usize>() as u32,
            ),
            "large",
        )
    } else {
        ((total_elems.div_ceil(block_dim.0 as usize) as u32, 1, 1), "generic")
    };

    let op_name = mini_op.name().to_lowercase();
    let tname = DeviceTensor::tname(lhs.datum_type())?;
    let kname = format!("binary_{op_name}_{variant}_{tname}");
    let func = cuda_context().load_pipeline(LibraryName::Binary, kname)?;

    let cfg = LaunchConfig { grid_dim, block_dim, shared_mem_bytes: 0 };

    let lhs_view = get_cuda_view(lhs);
    let rhs_view = get_cuda_view(rhs);
    let out_view = get_cuda_view(output);

    let mut launch_args = TractLaunchArgs::new(stream, &func);
    launch_args.push_view(&lhs_view);
    launch_args.push_view(&rhs_view);
    launch_args.push_view(&out_view);
    launch_args.push_slice_i32(&rhs_shape);
    launch_args.push_slice_i32(&out_shape);
    launch_args.push_slice_i32(&lhs_strides);
    launch_args.push_slice_i32(&rhs_strides);
    launch_args.push_slice_i32(&out_strides);

    launch_args.launch(cfg)?;

    Ok(())
}

pub fn cuda_bin_op_dispatch(
    mini_op: &dyn BinMiniOp,
    lhs: &DeviceTensor,
    rhs: &DeviceTensor,
    output: &DeviceTensor,
) -> TractResult<()> {
    crate::with_cuda_stream(|stream| dispatch_eval(stream, mini_op, lhs, rhs, output))
}

pub fn cuda_bin_op(mini_op: Box<dyn BinMiniOp>) -> tract_gpu::ops::binary::GpuBinOp {
    tract_gpu::ops::binary::GpuBinOp {
        backend_name: "Cuda",
        mini_op,
        dispatch: cuda_bin_op_dispatch,
    }
}

crate::register_cuda_op!(tract_core::ops::binary::TypedBinOp, |source, node, op| {
    rule_if!(is_supported(&*op.0, source.node_input_facts(node.id)?[0].datum_type));
    Ok(Some(Box::new(cuda_bin_op(op.0.clone()))))
});

#[cfg(test)]
mod tests {
    use tract_gpu::tensor::IntoDevice;

    use super::*;
    use crate::with_cuda_stream;
    use derive_new::new;
    use num_traits::AsPrimitive;
    use num_traits::Float;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use tract_core::internal::Tensor;

    fn test_case<F>(
        mini_op: &dyn BinMiniOp,
        shape: &[usize],
        offset: f32,
        scale: f32,
    ) -> TractResult<()>
    where
        F: Float + Datum,
        usize: AsPrimitive<f32>,
        f32: AsPrimitive<F>,
    {
        with_cuda_stream(|stream| {
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

            let out_dt = mini_op.result_datum_type(a.datum_type(), b.datum_type())?;
            let output = unsafe { DeviceTensor::uninitialized_dt(out_dt, shape)? };
            dispatch_eval(stream, mini_op, &a, &b, &output)?;
            stream.synchronize()?;

            let out = output.to_host()?.into_tensor();
            assert_eq!(out.shape(), shape);
            Ok(())
        })
    }

    use tract_core::ops::math;

    #[test]
    fn test_binary_add() -> TractResult<()> {
        test_case::<f32>(&math::Add, &[4, 4], 0.0, 1.0)?;
        test_case::<f16>(&math::Add, &[4, 4], 0.0, 1.0 / 100.0)?;
        Ok(())
    }

    #[test]
    fn test_binary_mul() -> TractResult<()> {
        test_case::<f32>(&math::Mul, &[4, 4], 0.0, 1.0)?;
        test_case::<f16>(&math::Mul, &[4, 4], 0.0, 1.0 / 100.0)?;
        Ok(())
    }

    #[test]
    fn test_binary_sub() -> TractResult<()> {
        test_case::<f32>(&math::Sub, &[4, 4], 0.0, 1.0)?;
        Ok(())
    }

    #[test]
    fn test_binary_min() -> TractResult<()> {
        test_case::<f32>(&math::Min, &[4, 4], 0.0, 1.0)?;
        Ok(())
    }

    #[test]
    fn test_binary_max() -> TractResult<()> {
        test_case::<f32>(&math::Max, &[4, 4], 0.0, 1.0)?;
        Ok(())
    }
}
