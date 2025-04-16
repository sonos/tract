use crate::encoder::EncoderExt;
use tract_gpu::tensor::GpuTensor;
use crate::{LibraryName, MetalContext};
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

    pub fn kernel_name(&self, dt: DatumType) -> Result<String> {
        ensure!(Self::is_supported_dt(dt), "Unsupport dt {:?} for metal copy  op", dt);
        let tname = GpuTensor::tname(dt)?;
        Ok(format!("array_ops::copy_unicast_{tname}"))
    }

    pub fn dispatch_eval(
        &self,
        context: &MetalContext,
        input: &GpuTensor,
        input_offset: usize,
        output: &GpuTensor,
    ) -> Result<()> {
        ensure!(input_offset % input.datum_type().size_of() == 0);
        ensure!(output.len() <= input.len() - input_offset);

        context.retain_tensor(input);
        context.retain_tensor(output);

        let kernel_name = self.kernel_name(input.datum_type())?;

        let pipeline =
            context.shared_context().load_pipeline(LibraryName::ArrayOps, &kernel_name)?;
        let command_buffer = context.command_buffer();
        command_buffer.encode(|encoder| {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_metal_tensor_with_offset(
                0,
                input,
                input_offset as _,
                metal::MTLResourceUsage::Read,
            );
            encoder.set_metal_tensor(1, output, metal::MTLResourceUsage::Write);

            let grid_size = MTLSize { width: output.len() as NSUInteger, height: 1, depth: 1 };
            let group_size = MTLSize { width: 1, height: 1, depth: 1 };
            encoder.dispatch_thread_groups(grid_size, group_size);
        });
        Ok(())
    }

    pub fn eval(
        &self,
        context: &MetalContext,
        input: &GpuTensor,
        input_offset: usize,
        output_shape: &[usize],
    ) -> Result<GpuTensor> {
        let output = unsafe { GpuTensor::uninitialized_dt(input.datum_type(), output_shape)? };
        self.dispatch_eval(context, input, input_offset, &output)?;
        context.wait_until_completed()?;
        Ok(output)
    }
}
