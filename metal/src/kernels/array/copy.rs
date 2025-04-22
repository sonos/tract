use crate::encoder::EncoderExt;
use crate::{LibraryName, MetalStream};
use derive_new::new;
use metal::{MTLSize, NSUInteger};
use std::fmt;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

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

    pub fn kernel_name(&self, dt: DatumType) -> TractResult<String> {
        ensure!(Self::is_supported_dt(dt), "Unsupport dt {:?} for metal copy  op", dt);
        let tname = DeviceTensor::tname(dt)?;
        Ok(format!("array_ops::copy_unicast_{tname}"))
    }

    pub fn dispatch_eval(
        &self,
        stream: &MetalStream,
        input: &DeviceTensor,
        input_offset: usize,
        output: &DeviceTensor,
    ) -> TractResult<()> {
        ensure!(input_offset % input.datum_type().size_of() == 0);
        ensure!(output.len() <= input.len() - input_offset);

        stream.retain_tensor(input);
        stream.retain_tensor(output);

        let kernel_name = self.kernel_name(input.datum_type())?;

        let pipeline = stream.load_pipeline(LibraryName::ArrayOps, &kernel_name)?;
        let command_buffer = stream.command_buffer();
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
        stream: &MetalStream,
        input: &DeviceTensor,
        input_offset: usize,
        output_shape: &[usize],
    ) -> TractResult<DeviceTensor> {
        let output = unsafe { DeviceTensor::uninitialized_dt(input.datum_type(), output_shape)? };
        self.dispatch_eval(stream, input, input_offset, &output)?;
        stream.wait_until_completed()?;
        Ok(output)
    }
}
