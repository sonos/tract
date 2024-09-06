use crate::MetalTensor;
use crate::{LibraryName, MetalContext};
use anyhow::bail;
use anyhow::Result;
use derive_new::new;
use metal::{MTLSize, NSUInteger};
use std::fmt;
use tract_core::internal::*;

#[derive(Debug, Clone, new, PartialEq, Eq, Hash)]
pub struct Memcpy;

impl fmt::Display for Memcpy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Memcpy {
    pub fn is_supported_dt(dt: DatumType) -> bool {
        Self::tname(dt).is_ok()
    }

    pub fn tname(dt: DatumType) -> Result<&'static str> {
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
            _ => bail!("Unsupport dt {:?} for metal array ops", dt),
        };
        Ok(tname)
    }

    pub fn kernel_name(&self, from_dt: DatumType) -> Result<String> {
        let tname = Self::tname(from_dt)?;
        Ok(format!("array_ops::copy_unicast_{tname}"))
    }

    pub fn dispatch_eval(
        &self,
        context: &MetalContext,
        input: &MetalTensor,
        input_offset: usize,
    ) -> Result<MetalTensor> {
        ensure!(input_offset % input.datum_type().size_of() == 0);
        let output = unsafe { MetalTensor::uninitialized_dt(input.datum_type(), input.shape())? };
        let kernel_name = self.kernel_name(input.datum_type())?;

        let input_buffer = input.metal();
        let output_buffer = output.metal();
        let pipeline =
            context.shared_context().load_pipeline(LibraryName::ArrayOps, &kernel_name)?;
        let command_buffer = context.command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(input_buffer), input_offset as NSUInteger);
        encoder.set_buffer(1, Some(output.metal()), 0);

        let grid_size = MTLSize { width: output.len() as NSUInteger, height: 1, depth: 1 };
        let group_size = MTLSize { width: 1, height: 1, depth: 1 };
        encoder.use_resource(input_buffer, metal::MTLResourceUsage::Read);
        encoder.use_resource(output_buffer, metal::MTLResourceUsage::Write);
        encoder.dispatch_thread_groups(grid_size, group_size);
        encoder.end_encoding();
        Ok(output)
    }

    pub fn eval(
        &self,
        context: &MetalContext,
        input: &MetalTensor,
        input_offset: usize,
    ) -> Result<MetalTensor> {
        let output = self.dispatch_eval(context, input, input_offset)?;
        context.wait_until_completed()?;
        Ok(output)
    }
}
