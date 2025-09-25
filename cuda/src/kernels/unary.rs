use std::fmt;

use cudarc::driver::{LaunchConfig, PushKernelArg};
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

use crate::context::{TractCudaStream, cuda_context};
use crate::kernels::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy)]
pub enum UnaryOps {
    Neg,
    Abs,
    Sqr,
    Sqrt,
    Rsqrt,
    Recip,
    Ceil,
    Floor,
    Round,
    RoundHalfToEven,
    Exp,
    Sigmoid,
    Sin,
    Sinh,
    Asin,
    Asinh,
    Cos,
    Cosh,
    Acos,
    Acosh,
    Tan,
    Tanh,
    Atan,
    Atanh,
    Erf,
    Ln,
    Silu,
}

impl fmt::Display for UnaryOps {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl UnaryOps {
    pub const ALL: [UnaryOps; 27] = [
        Self::Neg,
        Self::Abs,
        Self::Sqr,
        Self::Sqrt,
        Self::Rsqrt,
        Self::Recip,
        Self::Ceil,
        Self::Floor,
        Self::Round,
        Self::RoundHalfToEven,
        Self::Exp,
        Self::Sigmoid,
        Self::Sin,
        Self::Sinh,
        Self::Asin,
        Self::Asinh,
        Self::Cos,
        Self::Cosh,
        Self::Acos,
        Self::Acosh,
        Self::Tan,
        Self::Tanh,
        Self::Atan,
        Self::Atanh,
        Self::Erf,
        Self::Ln,
        Self::Silu,
    ];

    pub fn is_supported_dt(dt: DatumType) -> bool {
        matches!(dt, DatumType::F32 | DatumType::F16)
    }

    pub fn name(&self) -> Cow<'_, str> {
        format!("{self}").into()
    }
    pub fn all_functions() -> Vec<String> {
        Self::ALL
            .into_iter()
            .flat_map(|op| DeviceTensor::SUPPORTED_DT.into_iter().map(move |dt| (op, dt)))
            .flat_map(|(op, dt)| op.kernel_name(dt).into_iter())
            .collect()
    }

    pub fn float_only(&self) -> bool {
        matches!(
            self,
            Self::Exp
                | Self::Ln
                | Self::Sigmoid
                | Self::Sqr
                | Self::Rsqrt
                | Self::Sqrt
                | Self::Recip
                | Self::Cos
                | Self::Acos
                | Self::Acosh
                | Self::Cosh
                | Self::Sin
                | Self::Asin
                | Self::Asinh
                | Self::Sinh
                | Self::Tan
                | Self::Atan
                | Self::Atanh
                | Self::Tanh
                | Self::Erf
                | Self::Neg
                | Self::Abs
                | Self::RoundHalfToEven
        )
    }

    pub fn kernel_name(&self, dt: DatumType) -> TractResult<String> {
        if self.float_only() && !matches!(dt, DatumType::F16 | DatumType::F32) {
            bail!("Unsupported dt for Cuda element wise ops: {:?}", self);
        }

        ensure!(Self::is_supported_dt(dt), "Unsupported dt {:?} for Cuda Unary Op", dt);
        let name = if matches!(self, Self::RoundHalfToEven) {
            "rint".to_string()
        } else {
            self.name().to_lowercase()
        };
        Ok(format!("unary_{}_{}", name, DeviceTensor::tname(dt)?))
    }

    pub fn eval(
        &self,
        stream: &TractCudaStream,
        input: &DeviceTensor,
    ) -> TractResult<DeviceTensor> {
        let output = unsafe { DeviceTensor::uninitialized_dt(input.datum_type(), input.shape())? };
        self.dispatch_eval(stream, input, &output)?;
        stream.synchronize()?;

        Ok(output)
    }

    pub fn dispatch_eval(
        &self,
        stream: &TractCudaStream,
        input: &DeviceTensor,
        output: &DeviceTensor,
    ) -> TractResult<()> {
        ensure!(output.shape() == input.shape());
        ensure!(output.datum_type() == input.datum_type());

        let func = cuda_context()
            .load_pipeline(LibraryName::Unary, self.kernel_name(input.datum_type())?)?;

        let len = input.len();

        let i_view = get_cuda_view(input);
        let o_view = get_cuda_view(output);

        let cfg = LaunchConfig::for_num_elems(len as _);
        let mut launch_args = stream.launch_builder(&func);
        launch_args.arg(&i_view);
        launch_args.arg(&o_view);
        launch_args.arg(&len);

        unsafe { launch_args.launch(cfg) }?;

        Ok(())
    }
}
