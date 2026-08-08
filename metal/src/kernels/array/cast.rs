use crate::encoder::EncoderExt;

use crate::{LibraryName, MetalStream};
use derive_new::new;
use metal::{MTLSize, NSUInteger};
use std::fmt;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

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
                | DatumType::Bool
        )
    }

    pub fn kernel_name(&self, from_dt: DatumType, to_dt: DatumType) -> TractResult<String> {
        ensure!(
            Self::is_supported_dt(from_dt),
            "Unsupported from_dt {:?} for metal castop",
            from_dt
        );
        ensure!(Self::is_supported_dt(to_dt), "Unsupported to_dt {:?} for metal castop", to_dt);
        let from_tname = DeviceTensor::tname(from_dt)?;
        let to_tname = DeviceTensor::tname(to_dt)?;
        Ok(format!("array_ops::cast_{from_tname}_{to_tname}"))
    }

    pub fn eval(
        &self,
        stream: &MetalStream,
        input: &DeviceTensor,
        to_dt: DatumType,
    ) -> TractResult<DeviceTensor> {
        let output = unsafe { DeviceTensor::uninitialized_dt(to_dt, input.shape())? };
        self.dispatch_eval(stream, input, &output)?;
        stream.wait_until_completed()?;
        Ok(output)
    }

    pub fn dispatch_eval(
        &self,
        stream: &MetalStream,
        input: &DeviceTensor,
        output: &DeviceTensor,
    ) -> TractResult<()> {
        stream.retain_tensor(input);
        stream.retain_tensor(output);
        ensure!(
            input.shape() == output.shape(),
            "Cast I/O don't have the same shape in: {:?}, out: {:?}",
            input.shape(),
            output.shape()
        );

        let kernel_name = self.kernel_name(input.datum_type(), output.datum_type())?;

        let pipeline = stream.load_pipeline(LibraryName::ArrayOps, &kernel_name)?;
        let total = output.len() as u64;
        if total == 0 {
            return Ok(());
        }
        let mut group_width =
            (pipeline.max_total_threads_per_threadgroup() as u64).min(256).min(total);
        while total % group_width != 0 {
            group_width -= 1;
        }
        let grid_width = total / group_width;
        let command_buffer = stream.command_buffer();
        command_buffer.encode(|encoder| {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_metal_tensor(0, input, metal::MTLResourceUsage::Read);
            encoder.set_metal_tensor(1, output, metal::MTLResourceUsage::Write);

            let grid_size = MTLSize { width: grid_width as NSUInteger, height: 1, depth: 1 };
            let group_size = MTLSize { width: group_width as NSUInteger, height: 1, depth: 1 };
            encoder.dispatch_thread_groups(grid_size, group_size);
        });
        Ok(())
    }
}

pub fn metal_cast_dispatch(input: &DeviceTensor, output: &DeviceTensor) -> TractResult<()> {
    crate::with_metal_stream(|stream| Cast.dispatch_eval(stream, input, output))
}

crate::register_metal_op!(tract_core::ops::cast::Cast, |_source, _node, op| {
    Ok(crate::transform::metal_cast_new(op.to).map(|c| Box::new(c) as _))
});
