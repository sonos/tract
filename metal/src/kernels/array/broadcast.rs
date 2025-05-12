use crate::encoder::EncoderExt;
use crate::kernels::{utils, BroadcastKind};

use crate::kernels::utils::compute_broadcast_strides;
use crate::{LibraryName, MetalStream};
use anyhow::ensure;
use std::fmt;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MultiBroadcast;

impl fmt::Display for MultiBroadcast {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl MultiBroadcast {
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

    pub fn kernel_name(&self, dt: DatumType, broadcast_kind: BroadcastKind) -> TractResult<String> {
        ensure!(Self::is_supported_dt(dt), "Unsupport dt {:?} for metal broadcast  op", dt);
        let tname = DeviceTensor::tname(dt)?;
        let broadcast_name = broadcast_kind.name();
        Ok(format!("array_ops::copy_{broadcast_name}_{tname}"))
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

    pub fn dispatch_eval(
        &self,
        stream: &MetalStream,
        input: &DeviceTensor,
        input_offset: usize,
        output: &DeviceTensor,
    ) -> TractResult<()> {
        stream.retain_tensor(input);
        stream.retain_tensor(output);

        ensure!(input_offset % input.datum_type().size_of() == 0);
        ensure!(input.rank() <= output.rank(), "Input must have a rank lower or equal to output");

        let mut input_shape = vec![1; output.rank() - input.rank()];
        input_shape.extend(input.shape());
        let mut input_strides = vec![input.strides()[0]; output.rank() - input.rank()];
        input_strides.extend(input.strides());

        let broadcast_kind = BroadcastKind::from_rank(output.rank()).with_context(|| {
            format!(
                "Unsupported broadcast for broadcast op: (in: {:?}, out: {:?})",
                input.shape(),
                output.shape(),
            )
        })?;

        let kernel_name = self.kernel_name(input.datum_type(), broadcast_kind)?;

        let input_broadcast_strides =
            compute_broadcast_strides::<usize>(input_shape.as_slice(), input_strides.as_slice())?;

        let out_shape = output.shape();
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
            encoder.set_slice(1, &input_broadcast_strides);
            encoder.set_metal_tensor(2, output, metal::MTLResourceUsage::Write);
            encoder.set_slice(3, out_shape);
            encoder.set_slice(4, output.strides());

            let (grid_size, group_size) = utils::build_metal_grid_and_groups_for_el_wise_op(
                out_shape,
                pipeline.max_total_threads_per_threadgroup() as _,
            );

            encoder.dispatch_thread_groups(grid_size, group_size);
        });
        Ok(())
    }
}
