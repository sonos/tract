use super::BroadcastKind;
use crate::MetalTensor;
use crate::{LibraryName, MetalContext};
use anyhow::bail;
use anyhow::{ensure, Result};
use metal::{MTLSize, NSUInteger};
use std::fmt;
use tract_core::internal::*;

// #[derive(Debug, Clone, PartialEq, Eq, Hash)]
// pub enum ArrayOps {
//     Cast { to: DatumType },
//     MultiBroadcastTo { shape: ShapeFact },
// }

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MultiBroadcastCast;

impl fmt::Display for MultiBroadcastCast {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl MultiBroadcastCast {
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

    pub fn kernel_name(
        &self,
        from_dt: DatumType,
        to_dt: DatumType,
        broadcast_kind: BroadcastKind,
    ) -> Result<String> {
        let from_tname = Self::tname(from_dt)?;
        let to_tname = Self::tname(to_dt)?;

        let kbroadcast_name = match broadcast_kind {
            BroadcastKind::Unicast => "unicast",
            BroadcastKind::Nd1 => "nd1",
            BroadcastKind::Nd2 => "nd2",
            BroadcastKind::Nd3 => "nd3",
            BroadcastKind::Nd4 => "nd4",
            BroadcastKind::Nd5 => "nd5",
            _ => bail!("Unsupported broadcast kind {:?} for array ops", broadcast_kind),
        };

        Ok(format!("array_ops::broadcast_cast_{kbroadcast_name}_{from_tname}_{to_tname}"))
    }

    pub fn eval(
        &self,
        context: &MetalContext,
        input: &MetalTensor,
        output_dt: DatumType,
        output_shape: &[usize],
    ) -> Result<MetalTensor> {
        let output = unsafe { MetalTensor::uninitialized_dt(output_dt, output_shape)? };

        let broadcast_kind = if input.shape() == output_shape {
            BroadcastKind::Unicast
        } else if output.rank() == 1 {
            BroadcastKind::Nd1
        } else if output.rank() == 2 {
            BroadcastKind::Nd2
        } else if output.rank() == 3 {
            BroadcastKind::Nd3
        } else if output.rank() == 4 {
            BroadcastKind::Nd4
        } else if output.rank() == 5 {
            BroadcastKind::Nd5
        } else {
            bail!(
                "Unsupport broadcast for mem op: (in: {:?}, out: {:?})",
                input.shape(),
                output_shape,
            );
        };

        let kernel_name = self.kernel_name(input.datum_type(), output_dt, broadcast_kind)?;

        match broadcast_kind {
            BroadcastKind::Unicast => {
                let input_buffer = input.metal();
                let output_buffer = output.metal();
                let pipeline =
                    context.shared_context().load_pipeline(LibraryName::ArrayOps, &kernel_name)?;
                let command_buffer = context.command_buffer()?;
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
            }
            BroadcastKind::Nd1
            | BroadcastKind::Nd2
            | BroadcastKind::Nd3
            | BroadcastKind::Nd4
            | BroadcastKind::Nd5 => {
                ensure!(input.rank() == output.rank());
                let input_buffer = input.metal();
                let output_buffer = output.metal();
                let input_strides = input
                    .strides()
                    .iter()
                    .zip(input.shape())
                    .map(|(s, dim)| if *dim == 1 { 0 } else { *s as u32 })
                    .collect::<Vec<_>>();

                let output_shape = output.shape().iter().map(|d| *d as u32).collect::<Vec<_>>();

                let pipeline =
                    context.shared_context().load_pipeline(LibraryName::BinOps, &kernel_name)?;
                let command_buffer = context.command_buffer()?;
                let encoder = command_buffer.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&pipeline);
                encoder.set_buffer(0, Some(input_buffer), 0);
                encoder.set_bytes(
                    1,
                    (input_strides.len() * std::mem::size_of::<u32>()) as NSUInteger,
                    input_strides.as_ptr() as *const _,
                );
                encoder.set_buffer(2, Some(output.metal()), 0);
                encoder.set_bytes(
                    3,
                    (output_shape.len() * std::mem::size_of::<u32>()) as NSUInteger,
                    output_shape.as_ptr() as *const _,
                );

                let grid_size = MTLSize {
                    width: output_shape[output_shape.len() - 1] as NSUInteger,
                    height: output_shape[output_shape.len() - 2] as NSUInteger,
                    depth: (output_shape[..output_shape.len() - 2].iter().product::<u32>())
                        as NSUInteger,
                };

                let group_size = MTLSize { width: 1, height: 1, depth: 1 };
                encoder.use_resource(input_buffer, metal::MTLResourceUsage::Read);
                encoder.use_resource(output_buffer, metal::MTLResourceUsage::Write);
                encoder.dispatch_thread_groups(grid_size, group_size);
                encoder.end_encoding();
            }
            BroadcastKind::ByScalarLeft | BroadcastKind::ByScalarRight => {
                bail!("Unsupported broadcast kind {:?} for array ops", broadcast_kind)
            }
        }

        context.wait_until_completed()?;
        Ok(output)
    }
}
