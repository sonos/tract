use crate::kernels::{utils, BroadcastKind};
use crate::MetalTensor;
use crate::{LibraryName, MetalContext};
use anyhow::bail;
use anyhow::{ensure, Result};
use std::fmt;
use tract_core::internal::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MultiBroadcast;

impl fmt::Display for MultiBroadcast {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl MultiBroadcast {
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
            _ => bail!("Unsupport dt {:?} for metal memory ops", dt),
        };
        Ok(tname)
    }

    pub fn kernel_name(&self, dt: DatumType, broadcast_kind: BroadcastKind) -> Result<String> {
        let tname = Self::tname(dt)?;

        let broadcast_name = match broadcast_kind {
            BroadcastKind::Nd1 => "nd1",
            BroadcastKind::Nd2 => "nd2",
            BroadcastKind::Nd3 => "nd3",
            BroadcastKind::Nd4 => "nd4",
            BroadcastKind::Nd5 => "nd5",
            BroadcastKind::Nd6 => "nd6",
            _ => bail!("Unsupported broadcast kind {:?} for array ops", broadcast_kind),
        };

        Ok(format!("array_ops::broadcast_{broadcast_name}_{tname}"))
    }

    pub fn eval(
        &self,
        context: &MetalContext,
        input: &MetalTensor,
        input_offset: usize,
        output_shape: &[usize],
    ) -> Result<MetalTensor> {
        let output = self.dispatch_eval(context, input, input_offset, output_shape)?;
        context.wait_until_completed()?;
        Ok(output)
    }

    pub fn dispatch_eval(
        &self,
        context: &MetalContext,
        input: &MetalTensor,
        input_offset: usize,
        output_shape: &[usize],
    ) -> Result<MetalTensor> {
        ensure!(input_offset % input.datum_type().size_of() == 0);

        let output = unsafe { MetalTensor::uninitialized_dt(input.datum_type(), output_shape)? };
        ensure!(input.rank() <= output.rank(), "Input must have a rank lowe than output");

        let mut input_shape = vec![1; output.rank() - input.rank()];
        input_shape.extend(input.shape());
        let input_strides = Tensor::natural_strides(&input_shape);

        let broadcast_kind = BroadcastKind::from_rank(output.rank()).with_context(|| {
            anyhow!(
                "Unsupported broadcast for broadcast op: (in: {:?}, out: {:?})",
                input.shape(),
                output_shape
            )
        })?;

        let kernel_name = self.kernel_name(input.datum_type(), broadcast_kind)?;

        let input_broadcast_strides = crate::utils::compute_broadcast_strides::<u32>(
            input_shape.as_slice(),
            input_strides.as_slice(),
        )?;

        let output_shape = output.shape().iter().map(|d| *d as u32).collect::<Vec<_>>();

        let pipeline =
            context.shared_context().load_pipeline(LibraryName::ArrayOps, &kernel_name)?;
        let command_buffer = context.command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(input.metal()), input_offset as _);
        encoder.set_bytes(
            1,
            (input_broadcast_strides.len() * std::mem::size_of::<u32>()) as _,
            input_broadcast_strides.as_ptr() as *const _,
        );
        encoder.set_buffer(2, Some(output.metal()), 0);
        encoder.set_bytes(
            3,
            (output_shape.len() * std::mem::size_of::<u32>()) as _,
            output_shape.as_ptr() as *const _,
        );

        let grid_size = utils::build_metal_size_for_shape(output.shape());
        let group_size = utils::build_metal_size_with_ones();

        encoder.use_resource(input.metal(), metal::MTLResourceUsage::Read);
        encoder.use_resource(output.metal(), metal::MTLResourceUsage::Write);
        encoder.dispatch_thread_groups(grid_size, group_size);
        encoder.end_encoding();
        Ok(output)
    }
}
