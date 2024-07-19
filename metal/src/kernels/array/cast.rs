use crate::MetalTensor;
use crate::{LibraryName, MetalContext};
use anyhow::Result;
use derive_new::new;
use metal::{MTLSize, NSUInteger};
use std::fmt;
use tract_core::internal::*;

#[derive(Debug, Clone, new, PartialEq, Eq, Hash)]
pub struct Cast;

impl fmt::Display for Cast {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Cast {
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

    pub fn kernel_name(&self, from_dt: DatumType, to_dt: DatumType) -> Result<String> {
        ensure!(
            Self::is_supported_dt(from_dt),
            "Unsupported from_dt {:?} for metal cast  op",
            from_dt
        );
        ensure!(Self::is_supported_dt(to_dt), "Unsupported to_dt {:?} for metal cast  op", to_dt);
        let from_tname = MetalTensor::tname(from_dt)?;
        let to_tname = MetalTensor::tname(to_dt)?;
        Ok(format!("array_ops::cast_{from_tname}_{to_tname}"))
    }

    pub fn eval(
        &self,
        context: &MetalContext,
        input: &MetalTensor,
        to_dt: DatumType,
    ) -> Result<MetalTensor> {
        let o = self.dispatch_eval(context, input, to_dt)?;
        context.wait_until_completed()?;
        Ok(o)
    }

    pub fn dispatch_eval(
        &self,
        context: &MetalContext,
        input: &MetalTensor,
        to_dt: DatumType,
    ) -> Result<MetalTensor> {
        let output = unsafe { MetalTensor::uninitialized_dt(to_dt, input.shape())? };
        input.retain_until_completion();
        output.retain_until_completion();

        let kernel_name = self.kernel_name(input.datum_type(), to_dt)?;

        let input_buffer = input.metal();
        let output_buffer = output.metal();
        let pipeline =
            context.shared_context().load_pipeline(LibraryName::ArrayOps, &kernel_name)?;
        let command_buffer = context.command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(input_buffer), 0);
        encoder.set_buffer(1, Some(output.metal()), 0);

        let grid_size = MTLSize { width: output.len() as NSUInteger, height: 1, depth: 1 };
        let group_size = MTLSize { width: 1, height: 1, depth: 1 };
        encoder.use_resource(input_buffer, metal::MTLResourceUsage::Read);
        encoder.use_resource(output_buffer, metal::MTLResourceUsage::Write);
        encoder.dispatch_thread_groups(grid_size, group_size);
        encoder.end_encoding();
        Ok(output)
    }
}
