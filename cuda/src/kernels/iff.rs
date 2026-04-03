use cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg};
use std::fmt;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

use crate::context::{TractCudaStream, cuda_context};
use crate::kernels::launch_args::TractLaunchArgs;
use crate::kernels::utils::compute_broadcast_strides;
use crate::kernels::{LibraryName, MAX_THREADS, get_cuda_view};

static TERNARY_MAX_RANK: usize = 5;

#[derive(Debug, PartialEq)]
pub struct Iff;

impl fmt::Display for Iff {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Iff")
    }
}

impl Iff {
    pub fn name(&self) -> Cow<'_, str> {
        format!("{self}").into()
    }

    pub fn output_shape<D: DimLike>(&self, a: &[D], b: &[D], c: &[D]) -> TractResult<TVec<D>> {
        tract_core::broadcast::multi_broadcast(&[a, b, c])
            .with_context(|| format!("Error while broadcasting {a:?} {b:?} {c:?}"))
    }

    pub fn kernel_name(&self, dt: DatumType, variant: &str) -> TractResult<String> {
        Ok(format!("iff_{variant}_{}", tract_gpu::utils::BroadcastKind::copy_tname(dt)))
    }

    pub fn eval(
        &self,
        stream: &TractCudaStream,
        cond: &DeviceTensor,
        then_value: &DeviceTensor,
        else_value: &DeviceTensor,
    ) -> TractResult<DeviceTensor> {
        let out_shape = self.output_shape(cond.shape(), then_value.shape(), else_value.shape())?;
        ensure!(then_value.datum_type() == else_value.datum_type());
        let out_dt = then_value.datum_type();
        let output = unsafe { DeviceTensor::uninitialized_dt(out_dt, &out_shape)? };

        self.dispatch_eval(stream, cond, then_value, else_value, &output)?;

        stream.synchronize()?;
        Ok(output)
    }

    pub fn dispatch_eval(
        &self,
        stream: &TractCudaStream,
        cond: &DeviceTensor,
        then_value: &DeviceTensor,
        else_value: &DeviceTensor,
        output: &DeviceTensor,
    ) -> TractResult<()> {
        let inputs = [cond, then_value, else_value];
        let rank = *[cond.rank(), then_value.rank(), else_value.rank()].iter().max().unwrap();
        ensure!(rank <= TERNARY_MAX_RANK);

        let rank_pad = TERNARY_MAX_RANK - rank;
        let mut strides = [[0isize; TERNARY_MAX_RANK]; 3];
        let mut out_shape = [1usize; TERNARY_MAX_RANK];
        let mut out_strides = [0isize; TERNARY_MAX_RANK];

        for axis in 0..rank {
            out_shape[rank_pad + axis] = output.shape()[axis];
            out_strides[rank_pad + axis] = output.strides()[axis];
            for input in 0..3 {
                strides[input][rank_pad + axis] =
                    if inputs[input].shape()[axis] < output.shape()[axis] {
                        0
                    } else {
                        inputs[input].strides()[axis]
                    };
            }
        }

        let total_elems: usize = out_shape.iter().product();
        let block_dim = (128_u32, 1, 1);
        let (grid_dim, variant) =
        //     if out_shape[TERNARY_MAX_RANK - 1] >= 256 && total_elems >= 4096 {
        //     (
        //         (
        //             out_shape[TERNARY_MAX_RANK - 2] as u32,
        //             out_shape[TERNARY_MAX_RANK - 3] as u32,
        //             out_shape[..TERNARY_MAX_RANK - 3].iter().product::<usize>() as u32,
        //         ),
        //         "large",
        //     )
        // } else {
            ((total_elems.div_ceil(block_dim.0 as usize) as u32, 1, 1), "generic")
        // };
        ;

        let kernel_name = self.kernel_name(output.datum_type(), variant)?;
        let func = cuda_context().load_pipeline(LibraryName::Binary, kernel_name)?;

        let cfg = LaunchConfig { grid_dim, block_dim, shared_mem_bytes: 0 };

        let cond_view = get_cuda_view(cond);
        let then_view = get_cuda_view(then_value);
        let else_view = get_cuda_view(else_value);
        let o_view = get_cuda_view(output);

        let mut launch_args = TractLaunchArgs::new(stream, &func);
        launch_args.push_view(&cond_view);
        launch_args.push_view(&then_view);
        launch_args.push_view(&else_view);
        launch_args.push_view(&o_view);
        launch_args.push_slice_i32(&out_shape);
        for stride in &strides {
            launch_args.push_slice_i32(stride);
        }
        launch_args.push_slice_i32(&out_strides);

        launch_args.launch(cfg)?;

        Ok(())
    }
}

pub fn cuda_iff_dispatch(
    cond: &DeviceTensor,
    then_value: &DeviceTensor,
    else_value: &DeviceTensor,
    cond_strides: &[isize],
    then_strides: &[isize],
    else_strides: &[isize],
    output: &DeviceTensor,
    output_shape: &[usize],
    output_strides: &[isize],
) -> TractResult<()> {
    crate::with_cuda_stream(|stream| {
        let total_elems: usize = output_shape.iter().product();
        let block_dim = (128_u32, 1, 1);
        let grid_dim = (total_elems.div_ceil(block_dim.0 as usize) as u32, 1, 1);

        let kernel_name = format!(
            "iff_generic_{}",
            tract_gpu::utils::BroadcastKind::copy_tname(output.datum_type())
        );
        let func = cuda_context().load_pipeline(LibraryName::Binary, kernel_name)?;
        let cfg = LaunchConfig { grid_dim, block_dim, shared_mem_bytes: 0 };

        let cond_view = get_cuda_view(cond);
        let then_view = get_cuda_view(then_value);
        let else_view = get_cuda_view(else_value);
        let o_view = get_cuda_view(output);

        let mut launch_args = TractLaunchArgs::new(stream, &func);
        launch_args.push_view(&cond_view);
        launch_args.push_view(&then_view);
        launch_args.push_view(&else_view);
        launch_args.push_view(&o_view);
        launch_args.push_slice_i32(output_shape);
        launch_args.push_slice_i32(cond_strides);
        launch_args.push_slice_i32(then_strides);
        launch_args.push_slice_i32(else_strides);
        launch_args.push_slice_i32(output_strides);

        launch_args.launch(cfg)
    })
}

crate::register_cuda_op!(tract_core::ops::logic::Iff, |_source, _node, _op| {
    Ok(Some(Box::new(tract_gpu::ops::iff::GpuIff {
        backend_name: "Cuda",
        dispatch: cuda_iff_dispatch,
    })))
});
