use cudarc::driver::{CudaStream, DeviceRepr, LaunchConfig, PushKernelArg};
use derive_new::new;
use std::fmt;
use tract_core::internal::*;
use tract_core::ops::array::PadMode;
use tract_core::tract_data::itertools::Itertools;
use tract_gpu::tensor::DeviceTensor;

use crate::context::{TractCudaStream, cuda_context};
use crate::kernels::launch_args::LaunchArgsExt;
use crate::kernels::{LibraryName, get_cuda_view, get_sliced_cuda_view, launch_args};

static PAD_MAX_RANK: usize = 5;

#[derive(Debug, Clone, new, PartialEq, Eq, Hash)]
pub struct Pad;

impl fmt::Display for Pad {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl Pad {
    pub fn is_supported_dt(dt: DatumType) -> bool {
        matches!(
            dt,
            DatumType::F32
                | DatumType::F16
                | DatumType::U8
                | DatumType::U16
                | DatumType::U32
                | DatumType::U64
                | DatumType::I8
                | DatumType::I16
                | DatumType::I32
                | DatumType::I64
                | DatumType::Bool
        )
    }

    pub fn output_shape<D: DimLike>(i_shape: &[D], pads: &[(D, D)]) -> TractResult<TVec<D>> {
        let mut output_shape: TVec<D> = i_shape.to_vec().into();
        for i in 0..i_shape.len() {
            let left = pads[i].0.clone();
            let right = pads[i].1.clone();
            output_shape[i] = output_shape[i].clone() + left + right;
        }
        Ok(output_shape)
    }

    pub fn kernel_name(dt: DatumType) -> TractResult<String> {
        ensure!(Self::is_supported_dt(dt), "Unsupported dt {:?} for cuda pad op", dt);
        let tname = DeviceTensor::tname(dt)?;
        Ok(format!("pad_constant_{tname}"))
    }

    fn dispatch_eval_constant_t<T: Datum + DeviceRepr>(
        stream: &TractCudaStream,
        input: &DeviceTensor,
        output: &DeviceTensor,
        pads_before: TVec<usize>,
        val: &Tensor,
    ) -> TractResult<()> {
        ensure!(val.datum_type() == input.datum_type());
        let rank = input.rank();
        ensure!(rank <= PAD_MAX_RANK);

        let kernel_name = Self::kernel_name(input.datum_type())?;

        let rank = input.rank();
        let i_view = get_cuda_view(input);
        let o_view = get_cuda_view(output);
        let fill_value = val.to_scalar::<T>()?;
        let mut i_shape = input.shape().to_vec();
        let mut o_shape = output.shape().to_vec();
        let mut i_strides = input.strides().to_vec();
        let mut pads_before = pads_before;

        for _ in 0..(PAD_MAX_RANK - rank) {
            i_shape.push(1);
            o_shape.push(1);
            i_strides.push(0);
            pads_before.push(0);
        }

        let len = output.len();
        let func = cuda_context().load_pipeline(LibraryName::Array, kernel_name)?;
        let mut launch_args = stream.launch_builder(&func);
        launch_args.arg(&i_view);
        launch_args.arg(&o_view);
        launch_args.set_slice(&i_shape);
        launch_args.set_slice(&o_shape);
        launch_args.set_slice(&i_strides);
        launch_args.set_slice(&pads_before);
        launch_args.arg(fill_value);
        launch_args.arg(&len);

        let cfg = LaunchConfig::for_num_elems(len as _);
        unsafe { launch_args.launch(cfg) };

        Ok(())
    }

    pub fn dispatch_eval(
        &self,
        stream: &TractCudaStream,
        input: &DeviceTensor,
        output: &DeviceTensor,
        pads_before: TVec<usize>,
        mode: PadMode,
    ) -> TractResult<()> {
        if let PadMode::Constant(val) = mode {
            dispatch_numbers!(Self::dispatch_eval_constant_t(input.datum_type())(
                stream,
                input,
                output,
                pads_before,
                &val
            ))
        } else {
            bail!("Unsupported PadMode")
        }
    }

    pub fn eval(
        &self,
        stream: &TractCudaStream,
        input: &DeviceTensor,
        pads: Vec<(usize, usize)>,
        mode: PadMode,
    ) -> TractResult<DeviceTensor> {
        let output = unsafe {
            DeviceTensor::uninitialized_dt(
                input.datum_type(),
                &Self::output_shape(input.shape(), &pads)?,
            )?
        };
        let before_pads = pads.iter().map(|(bef, _)| *bef).collect_vec().into();
        self.dispatch_eval(stream, input, &output, before_pads, mode)?;
        stream.synchronize()?;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use std::f32::INFINITY;

    use crate::context::CUDA_STREAM;

    use super::*;
    use tract_core::internal::Tensor;
    use tract_core::ops::array::{Pad, PadMode};
    use tract_gpu::tensor::IntoDevice;

    fn run_test<T>(
        in_shape: &[usize],
        padding: Vec<(usize, usize)>,
        val: Arc<Tensor>,
    ) -> TractResult<()>
    where
        T: Datum + Copy + From<u8>,
    {
        CUDA_STREAM.with(|stream| {
            let num_elements = in_shape.iter().product();

            let a = Tensor::from_shape(
                &in_shape,
                &(0..num_elements).map(|f| T::from(f as u8)).collect::<Vec<_>>(),
            )?;

            let cpu_output = Pad { pads: padding.clone(), mode: PadMode::Constant(val.clone()) }
                .eval_with_session(0, &SessionState::default(), tvec![a.clone().into_tvalue()])?;

            let a_cuda = a.clone().into_device()?;
            let mut session_state = SessionState::default();
            let cuda_output = Pad.eval(stream, &a_cuda, padding, PadMode::Constant(val))?;

            cuda_output
                .to_host()?
                .into_tensor()
                .close_enough(&cpu_output[0], Approximation::Exact)?;
            Ok(())
        })
    }

    #[test]
    fn test_pad() -> TractResult<()> {
        run_test::<f32>(&[1, 1], vec![(0, 0), (0, 0)], tensor0(1f32).into())?;
        run_test::<f16>(&[1, 1], vec![(1, 1), (1, 1)], tensor0(f16::from_f32(1.0)).into())?;
        run_test::<u32>(&[3, 4], vec![(1, 3), (2, 0)], tensor0(1u32).into())?;
        run_test::<i32>(&[1, 2, 3], vec![(0, 2), (1, 1), (2, 0)], tensor0(1i32).into())?;
        run_test::<u8>(&[3, 2], vec![(0, 0), (0, 4)], tensor0(1u8).into())?;
        run_test::<i16>(&[2, 4], vec![(4, 0), (0, 0)], tensor0(1i16).into())?;
        run_test::<f32>(
            &[2, 4, 1, 4, 2],
            vec![(4, 2), (0, 1), (3, 1), (2, 0), (1, 3)],
            tensor0(INFINITY).into(),
        )?;
        Ok(())
    }
}
