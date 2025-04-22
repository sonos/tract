use crate::encoder::EncoderExt;
use crate::{LibraryName, MetalStream};
use anyhow::bail;
use metal::{MTLSize, NSUInteger};
use std::fmt;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy)]
pub enum ElementWiseOps {
    Abs,
    Exp,
    Ln,
    Sigmoid,
    Square,
    Sqrt,
    Rsqrt,
    Recip,
    Ceil,
    Floor,
    Round,
    RoundHalfToEven,
    Cos,
    Acos,
    Acosh,
    Cosh,
    Sin,
    Asin,
    Asinh,
    Sinh,
    Tan,
    Atan,
    Atanh,
    Tanh,
    Erf,
    Neg,
}

impl fmt::Display for ElementWiseOps {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl ElementWiseOps {
    pub const ALL: [ElementWiseOps; 26] = [
        Self::Abs,
        Self::Exp,
        Self::Ln,
        Self::Sigmoid,
        Self::Square,
        Self::Sqrt,
        Self::Rsqrt,
        Self::Recip,
        Self::Ceil,
        Self::Floor,
        Self::Round,
        Self::RoundHalfToEven,
        Self::Cos,
        Self::Acos,
        Self::Acosh,
        Self::Cosh,
        Self::Sin,
        Self::Asin,
        Self::Asinh,
        Self::Sinh,
        Self::Tan,
        Self::Atan,
        Self::Atanh,
        Self::Tanh,
        Self::Erf,
        Self::Neg,
    ];

    pub fn name(&self) -> Cow<str> {
        format!("{}", self).into()
    }

    pub fn all_functions() -> Vec<String> {
        Self::ALL
            .into_iter()
            .flat_map(|op| DeviceTensor::SUPPORTED_DT.into_iter().map(move |dt| (op, dt)))
            .flat_map(|(op, dt)| op.kernel_name(dt, false).into_iter())
            .collect()
    }

    pub fn validation(&self) -> Validation {
        Validation::Accurate
    }

    pub fn float_only(&self) -> bool {
        matches!(
            self,
            Self::Exp
                | Self::Ln
                | Self::Sigmoid
                | Self::Square
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
        )
    }

    pub fn kernel_name(&self, dt: DatumType, in_place: bool) -> TractResult<String> {
        if self.float_only() && !matches!(dt, DatumType::F32 | DatumType::F16) {
            bail!("Unsupport dt for metal element wise ops: {:?}", self);
        }

        ensure!(Self::is_supported_dt(dt), "Unsupport dt {:?} for metal element wise ops", dt);

        let tname = DeviceTensor::tname(dt)?;

        let kname = match self {
            Self::Abs => "abs",
            Self::Exp => "exp",
            Self::Ln => "ln",
            Self::Sigmoid => "sigmoid",
            Self::Square => "square",
            Self::Rsqrt => "rsqrt",
            Self::Sqrt => "sqrt",
            Self::Recip => "recip",
            Self::Ceil => "ceil",
            Self::Floor => "floor",
            Self::Round => "round",
            Self::RoundHalfToEven => "round_half_to_even",
            Self::Cos => "cos",
            Self::Acos => "acos",
            Self::Acosh => "acosh",
            Self::Cosh => "cosh",
            Self::Sin => "sin",
            Self::Asin => "asin",
            Self::Asinh => "asinh",
            Self::Sinh => "sinh",
            Self::Tan => "tan",
            Self::Atan => "atan",
            Self::Atanh => "atanh",
            Self::Tanh => "tanh",
            Self::Erf => "erf",
            Self::Neg => "neg",
        };

        if in_place {
            Ok(format!("element_wise_ops::{kname}_in_place_{tname}"))
        } else {
            Ok(format!("element_wise_ops::{kname}_out_of_place_{tname}"))
        }
    }

    pub fn dispatch_eval(
        &self,
        stream: &MetalStream,
        input: &DeviceTensor,
        output: &DeviceTensor,
    ) -> TractResult<()> {
        stream.retain_tensor(input);
        stream.retain_tensor(output);

        ensure!(output.shape() == input.shape() && output.datum_type() == input.datum_type());

        let kernel_name = self.kernel_name(input.datum_type(), false)?;

        let pipeline = stream.load_pipeline(LibraryName::ElementWiseOps, &kernel_name)?;
        let command_buffer = stream.command_buffer();
        command_buffer.encode(|encoder| {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_metal_tensor(0, input, metal::MTLResourceUsage::Read);
            encoder.set_metal_tensor(1, output, metal::MTLResourceUsage::Write);

            let grid_size = MTLSize { width: output.len() as NSUInteger, height: 1, depth: 1 };
            let group_size = MTLSize { width: 1, height: 1, depth: 1 };
            encoder.dispatch_thread_groups(grid_size, group_size);
        });
        Ok(())
    }

    pub fn eval(&self, stream: &MetalStream, a: &DeviceTensor) -> TractResult<DeviceTensor> {
        let output = unsafe { DeviceTensor::uninitialized_dt(a.datum_type(), a.shape())? };
        self.dispatch_eval(stream, a, &output)?;
        stream.wait_until_completed()?;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use crate::utils::with_borrowed_metal_stream;

    use super::*;
    use num_traits::Zero;
    use rand::Rng;
    use tract_gpu::tensor::IntoDevice;

    fn reference<F: Datum>(a: &Tensor, ca: impl Fn(&mut F, &F)) -> TractResult<Tensor> {
        let mut out = unsafe { Tensor::uninitialized_dt(a.datum_type(), a.shape())? };
        let a_view = a.to_array_view::<F>()?;
        let mut c = out.to_array_view_mut::<F>()?;
        tract_core::ndarray::Zip::from(&mut c).and_broadcast(a_view).for_each(ca);
        Ok(out)
    }

    fn run_test_case<F: Datum + Zero>(
        op: ElementWiseOps,
        a_shape: &[usize],
        neg: bool,
        ca: impl Fn(&mut F, &F),
    ) -> TractResult<()> {
        with_borrowed_metal_stream(|stream| {
            let a_len = a_shape.iter().product::<usize>();
            let mut rng = rand::thread_rng();
            let a = Tensor::from_shape(
                a_shape,
                &(0..a_len)
                    .map(|_f| {
                        if neg {
                            rng.gen_range(-10.0f32..10.0)
                        } else {
                            rng.gen_range(0.0f32..10.0)
                        }
                    })
                    .collect::<Vec<_>>(),
            )?
            .into_device()?;
            let output = op.eval(stream, &a)?;
            let ref_output = reference::<F>(&a.to_host()?.into_tensor(), ca)?;
            assert!(ref_output
                .close_enough(&output.to_host()?.into_tensor(), Approximation::Close)
                .is_ok());
            Ok(())
        })
    }

    #[test]
    fn test_element_wise() -> TractResult<()> {
        run_test_case::<f32>(ElementWiseOps::Abs, &[4, 4], true, |c, a| *c = a.abs())?;
        run_test_case::<f32>(ElementWiseOps::Ln, &[4, 4], false, |c, a| *c = a.ln())?;

        Ok(())
    }
}
