use crate::MetalTensor;
use crate::{LibraryName, MetalContext};
use anyhow::bail;
use anyhow::Result;
use metal::{MTLSize, NSUInteger};
use std::fmt;
use tract_core::internal::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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
    pub fn name(&self) -> Cow<str> {
        format!("{}", self).into()
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
        )
    }

    pub fn kernel_name(&self, dt: DatumType, in_place: bool) -> Result<String> {
        if self.float_only() && !matches!(dt, DatumType::F32 | DatumType::F16) {
            bail!("Unsupport dt for metal binary ops: {:?}", self);
        }
        let tname = match dt {
            DatumType::F32 => "f32",
            DatumType::F16 => "f16",
            DatumType::U8 => "u8",
            DatumType::U16 => "u16",
            DatumType::U32 => "u32",
            DatumType::U64 => "u64",
            DatumType::I8 => "i8",
            DatumType::I16 => "i16",
            DatumType::I32 => "i32",
            DatumType::I64 => "i64",
            DatumType::Bool => "bool",
            _ => bail!("Unsupport dt for metal binary ops: {:?}", self),
        };

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

    pub fn eval(&self, context: &MetalContext, a: &MetalTensor) -> Result<MetalTensor> {
        let output = unsafe { MetalTensor::uninitialized_dt(a.datum_type(), a.shape())? };
        let kernel_name = self.kernel_name(a.datum_type(), false)?;

        let a_buffer = a.metal();
        let output_buffer = output.metal();
        let pipeline =
            context.shared_context().load_pipeline(LibraryName::ElementWiseOps, &kernel_name)?;
        let command_buffer = context.command_buffer()?;
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(a_buffer), 0);
        encoder.set_buffer(1, Some(output_buffer), 0);

        let grid_size = MTLSize { width: output.len() as NSUInteger, height: 1, depth: 1 };
        let group_size = MTLSize { width: 1, height: 1, depth: 1 };
        encoder.use_resource(a_buffer, metal::MTLResourceUsage::Read);
        encoder.use_resource(output_buffer, metal::MTLResourceUsage::Write);
        encoder.dispatch_thread_groups(grid_size, group_size);
        encoder.end_encoding();

        context.wait_until_completed()?;
        Ok(output)
    }
}
