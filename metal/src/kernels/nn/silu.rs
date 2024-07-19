use crate::{LibraryName, MetalContext, MetalTensor};
use anyhow::Result;
use metal::MTLSize;
use tract_core::internal::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Silu;

impl Silu {
    pub fn is_supported_dt(dt: DatumType) -> bool {
        Self::tname(dt).is_ok()
    }

    pub fn tname(dt: DatumType) -> Result<&'static str> {
        let tname = match dt {
            DatumType::F32 => "f32",
            DatumType::F16 => "f16",
            _ => bail!("Unsupport dt {:?} for reducer op", dt),
        };
        Ok(tname)
    }

    pub fn kernel_name(&self, dt: DatumType) -> Result<String> {
        let tname = Self::tname(dt)?;
        Ok(format!("nn_ops::silu_{tname}"))
    }

    pub fn eval(&self, context: &MetalContext, input: &MetalTensor) -> Result<MetalTensor> {
        let o = self.dispatch_eval(context, input)?;
        context.wait_until_completed()?;
        Ok(o)
    }

    pub fn dispatch_eval(&self, context: &MetalContext, a: &MetalTensor) -> Result<MetalTensor> {
        a.retain_until_completion();

        let output = unsafe { MetalTensor::uninitialized_dt(a.datum_type(), a.shape())? };
        output.retained_until_completion();

        let kernel_name = self.kernel_name(a.datum_type())?;

        let a_buffer = a.metal();
        let output_buffer = output.metal();
        let pipeline = context.shared_context().load_pipeline(LibraryName::NNOps, &kernel_name)?;
        let command_buffer = context.command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(a_buffer), 0);
        encoder.set_buffer(1, Some(output_buffer), 0);

        let grid_size = MTLSize { width: output.len() as _, height: 1, depth: 1 };
        let group_size = MTLSize { width: 1, height: 1, depth: 1 };
        encoder.use_resource(a_buffer, metal::MTLResourceUsage::Read);
        encoder.use_resource(output_buffer, metal::MTLResourceUsage::Write);
        encoder.dispatch_thread_groups(grid_size, group_size);
        encoder.end_encoding();
        Ok(output)
    }
}
