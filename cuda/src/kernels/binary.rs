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

/// The two operands and the output of one binary, as the kernel addresses them:
/// a shape and a stride per axis, an operand broadcast along an axis holding a
/// stride of zero there.
struct BinaryGeometry {
    lhs_shape: TVec<usize>,
    rhs_shape: TVec<usize>,
    out_shape: TVec<usize>,
    lhs_strides: TVec<isize>,
    rhs_strides: TVec<isize>,
    out_strides: TVec<isize>,
}

impl BinaryGeometry {
    /// Merges `axis` with the one after it, if the three tensors all walk the
    /// pair as one run: either the outer stride is the inner one times its
    /// extent, or the axis is broadcast on both halves and stays broadcast.
    /// Answers whether it did.
    fn merge(&mut self, axis: usize) -> bool {
        let inner = self.out_shape[axis + 1];
        let runs = |strides: &TVec<isize>| {
            strides[axis] == strides[axis + 1] * inner as isize
                || (strides[axis] == 0 && strides[axis + 1] == 0)
        };
        if !runs(&self.lhs_strides) || !runs(&self.rhs_strides) || !runs(&self.out_strides) {
            return false;
        }
        for (shape, strides) in [
            (&mut self.lhs_shape, &mut self.lhs_strides),
            (&mut self.rhs_shape, &mut self.rhs_strides),
            (&mut self.out_shape, &mut self.out_strides),
        ] {
            // An operand broadcast along both halves is broadcast along the
            // merged axis, and says so with an extent of one.
            shape[axis] = if strides[axis] == 0 && strides[axis + 1] == 0 {
                1
            } else {
                shape[axis] * shape[axis + 1]
            };
            shape.remove(axis + 1);
            strides[axis] = strides[axis + 1];
            strides.remove(axis + 1);
        }
        true
    }
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

    let base_l_shape = lhs.shape();
    let base_r_shape = rhs.shape();
    let mut geo = BinaryGeometry {
        lhs_shape: base_l_shape.into(),
        rhs_shape: base_r_shape.into(),
        out_shape: output.shape().into(),
        lhs_strides: lhs.strides().into(),
        rhs_strides: rhs.strides().into(),
        out_strides: output.strides().into(),
    };
    for i in 0..rank {
        // A tensor of extent one against a longer operand is read at the same
        // place for every index of that axis.
        if base_l_shape[i] == 1 && base_r_shape[i] != 1 {
            geo.lhs_strides[i] = 0;
        }
        if base_r_shape[i] == 1 && base_l_shape[i] != 1 {
            geo.rhs_strides[i] = 0;
        }
    }
    // The kernel addresses BINARY_MAX_RANK axes. A deeper shape reaches it by
    // merging the adjacent axes it can, which is exact and costs nothing: the
    // pair is one axis of the same elements in the same order.
    while geo.out_shape.len() > BINARY_MAX_RANK {
        let merged = (0..geo.out_shape.len() - 1).find(|&axis| geo.merge(axis));
        ensure!(
            merged.is_some(),
            "Binary of rank {} has no two adjacent axes to merge into {BINARY_MAX_RANK}: \
             lhs {:?} rhs {:?}",
            geo.out_shape.len(),
            lhs.shape(),
            rhs.shape(),
        );
    }
    let rank = geo.out_shape.len();

    let rank_offset = BINARY_MAX_RANK - rank;
    let mut lhs_shape = [1usize; BINARY_MAX_RANK];
    let mut rhs_shape = [1usize; BINARY_MAX_RANK];
    let mut out_shape = [1usize; BINARY_MAX_RANK];
    let mut lhs_strides = [0isize; BINARY_MAX_RANK];
    let mut rhs_strides = [0isize; BINARY_MAX_RANK];
    let mut out_strides = [0isize; BINARY_MAX_RANK];
    for i in 0..rank {
        let dst = rank_offset + i;
        lhs_shape[dst] = geo.lhs_shape[i];
        rhs_shape[dst] = geo.rhs_shape[i];
        out_shape[dst] = geo.out_shape[i];
        lhs_strides[dst] = geo.lhs_strides[i];
        rhs_strides[dst] = geo.rhs_strides[i];
        out_strides[dst] = geo.out_strides[i];
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
    tract_gpu::ops::binary::GpuBinOp::new(mini_op, "Cuda", cuda_bin_op_dispatch)
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
