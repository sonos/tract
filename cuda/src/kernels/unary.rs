use std::fmt;

use cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg};
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

use crate::context::cuda_context;
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
    Silu
}

impl fmt::Display for UnaryOps {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
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
        Self::Silu
    ];

    pub fn is_supported_dt(dt: DatumType) -> bool {
        matches!(dt, DatumType::F32 | DatumType::F16)
    }

    pub fn name(&self) -> Cow<str> {
        format!("{}", self).into()
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

    pub fn eval(&self, stream: &CudaStream, input: &DeviceTensor) -> TractResult<DeviceTensor> {
        let output = unsafe { DeviceTensor::uninitialized_dt(input.datum_type(), input.shape())? };
        self.dispatch_eval(stream, input, &output)?;
        stream.synchronize()?;

        Ok(output)
    }

    pub fn dispatch_eval(
        &self,
        stream: &CudaStream,
        input: &DeviceTensor,
        output: &DeviceTensor,
    ) -> TractResult<()> {
        ensure!(output.shape() == input.shape());
        ensure!(output.datum_type() == input.datum_type());

        let func = cuda_context()
            .load_pipeline(LibraryName::UnaryOps, self.kernel_name(input.datum_type())?)?;

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

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::{AsPrimitive, Float};
    use proptest::prelude::*;
    use proptest::collection::vec;
    use tract_core::ndarray::ArrayD;
    use tract_core::ops::element_wise::ElementWiseOp;
    use tract_gpu::tensor::IntoDevice;
    use tract_transformers::ops::silu::Silu;

    use crate::context::CUDA_STREAM;

    #[derive(Debug)]
    pub struct UnaryOpProblem<F: Datum + Float>
    where
        F: Datum + Float,
        usize: AsPrimitive<F>,
        f32: AsPrimitive<F>,
    {   
        pub op: UnaryOps,
        pub input: ArrayD<F>,
    }

    impl<F> Arbitrary for UnaryOpProblem<F>
    where
        F: Datum + Float,
        usize: AsPrimitive<F>,
        f32: AsPrimitive<F>,
    {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
            let shape_strategy = prop::collection::vec(1usize..=5, 0..=4);

            shape_strategy
                .prop_flat_map(|shape| {
                    let len = shape.iter().product::<usize>();
                    let input = vec((-10i8..=10i8).prop_map(|i| F::from(i).unwrap() / F::from(2).unwrap()), len..=len)
                        .prop_map(move |vec| ArrayD::from_shape_vec(shape.to_vec(), vec).unwrap())
                        .boxed();

                    let op_strategy = prop::sample::select(UnaryOps::ALL.to_vec());

                    (input, op_strategy)
                })
                .prop_map(|(input, op)| UnaryOpProblem {
                    input,
                    op,
                })
                .boxed()
        }
    }

    impl<F> UnaryOpProblem<F>
    where
        F: Datum + Float + std::ops::AddAssign,
        usize: AsPrimitive<F>,
        f32: AsPrimitive<F>,
    {
        pub fn reference(&self) -> TractResult<Tensor> {
            let cpu_output = if self.op == UnaryOps::Silu {
                Silu.eval(tvec![self.input.clone().into_tvalue()])?.remove(0).into_tensor()
            }
            else {
                let cpu_op: Box<dyn ElementWiseMiniOp> = match self.op {
                    UnaryOps::Neg => Box::new(tract_core::ops::math::Neg {}),
                    UnaryOps::Abs => Box::new(tract_core::ops::math::Abs {}),
                    UnaryOps::Sqr => Box::new(tract_core::ops::math::Square {}),
                    UnaryOps::Sqrt => Box::new(tract_core::ops::math::Sqrt {}),
                    UnaryOps::Rsqrt => Box::new(tract_core::ops::math::Rsqrt {}),
                    UnaryOps::Recip => Box::new(tract_core::ops::math::Recip {}),
                    UnaryOps::Ceil => Box::new(tract_core::ops::math::Ceil {}),
                    UnaryOps::Floor => Box::new(tract_core::ops::math::Floor {}),
                    UnaryOps::Round => Box::new(tract_core::ops::math::Round {}),
                    UnaryOps::RoundHalfToEven => Box::new(tract_core::ops::math::RoundHalfToEven {}),
                    UnaryOps::Exp => Box::new(tract_core::ops::math::Exp {}),
                    UnaryOps::Sigmoid => Box::new(tract_core::ops::nn::Sigmoid {}),
                    UnaryOps::Sin => Box::new(tract_core::ops::math::Sin {}),
                    UnaryOps::Sinh => Box::new(tract_core::ops::math::Sinh {}),
                    UnaryOps::Asin => Box::new(tract_core::ops::math::Asin {}),
                    UnaryOps::Asinh => Box::new(tract_core::ops::math::Asinh {}),
                    UnaryOps::Cos => Box::new(tract_core::ops::math::Cos {}),
                    UnaryOps::Cosh => Box::new(tract_core::ops::math::Cosh {}),
                    UnaryOps::Acos => Box::new(tract_core::ops::math::Acos {}),
                    UnaryOps::Acosh => Box::new(tract_core::ops::math::Acosh {}),
                    UnaryOps::Tan => Box::new(tract_core::ops::math::Tan {}),
                    UnaryOps::Tanh => Box::new(tract_core::ops::math::Tanh {}),
                    UnaryOps::Atan => Box::new(tract_core::ops::math::Atan {}),
                    UnaryOps::Atanh => Box::new(tract_core::ops::math::Atanh {}),
                    UnaryOps::Erf => Box::new(tract_core::ops::math::Erf {}),
                    UnaryOps::Ln => Box::new(tract_core::ops::math::Ln {}),
                    _ => bail!("Could not convert to CPU op")
                };
                ElementWiseOp(cpu_op, None).eval(tvec![self.input.clone().into_tvalue()])?.remove(0).into_tensor()
            };

            Ok(cpu_output)
        }

        pub fn run(&self) -> TractResult<Tensor> {
            CUDA_STREAM.with(|stream| {
                let metal_output = self.op.eval(stream, &self.input.clone().into_tensor().into_device()?)?;
                Ok(metal_output.to_host()?.into_tensor())
            })
        }
    }

    proptest::proptest! {
        #[test]
        fn unary_prop_f32(pb in any::<UnaryOpProblem<f32>>()) {
            fn run(pb: UnaryOpProblem<f32>) -> TractResult<()> {
                let out = pb.run()?;
                let reference = pb.reference()?;

                out.close_enough(&reference, Approximation::Approximate)
                   .with_context(|| format!("Cpu: {:?}, Cuda: {:?}",  reference.dump(true), out.dump(true)))
            }
            run(pb).map_err(|e| TestCaseError::Fail(format!("{:?}", e).into()))?;
        }

        #[test]
        fn unary_prop_f16(pb in any::<UnaryOpProblem<f16>>()) {
            fn run(pb: UnaryOpProblem<f16>) -> TractResult<()> {
                let out = pb.run()?;
                let reference = pb.reference()?;

                out.close_enough(&reference, Approximation::VeryApproximate)
                   .with_context(|| format!("Cpu: {:?}, Cuda: {:?}", reference.dump(true), out.dump(true)))
            }

            run(pb).map_err(|e| TestCaseError::Fail(format!("{:?}", e).into()))?;
        }
    }
}
