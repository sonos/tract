use crate::kernels::utils;
use crate::MetalTensor;
use crate::{LibraryName, MetalContext};
use anyhow::{ensure, Result};
use metal::MTLSize;
use std::fmt;
use tract_core::internal::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ApplyRope;

impl fmt::Display for ApplyRope {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl ApplyRope {
    pub fn is_supported_dt(dt: DatumType) -> bool {
        matches!(dt, DatumType::F32)
    }

    pub fn kernel_name(&self, dt: DatumType) -> Result<String> {
        ensure!(Self::is_supported_dt(dt), "Unsupport dt {:?} for metal apply rope", dt);
        let tname = MetalTensor::tname(dt)?;
        Ok(format!("nn_ops::apply_rope_nd2_{tname}"))
    }

    pub fn eval(
        &self,
        context: &MetalContext,
        input: &MetalTensor,
        cos: &MetalTensor,
        sin: &MetalTensor,
    ) -> Result<MetalTensor> {
        let output = self.dispatch_eval(context, input, cos, sin)?;
        context.wait_until_completed()?;
        Ok(output)
    }

    pub fn dispatch_eval(
        &self,
        context: &MetalContext,
        input: &MetalTensor,
        cos: &MetalTensor,
        sin: &MetalTensor,
    ) -> Result<MetalTensor> {
        ensure!(input.datum_type() == cos.datum_type() && input.shape() == cos.shape());
        ensure!(input.datum_type() == sin.datum_type() && input.shape() == sin.shape());

        let output = unsafe { MetalTensor::uninitialized_dt(input.datum_type(), input.shape())? };

        input.retain_until_completion();
        output.retain_until_completion();

        let shape_nd2 = utils::reshape_to_rank_2(input.shape(), input.rank() - 1);
        ensure!(
            shape_nd2[1] % 2 == 0,
            "Rotate half required most inner dimension to be a multiple of 2: {:?}",
            input.shape()
        );
        let strides_nd2 = Tensor::natural_strides(&shape_nd2);

        let kernel_name = self.kernel_name(input.datum_type())?;

        let pipeline =
            context.shared_context().load_pipeline(LibraryName::ArrayOps, &kernel_name)?;
        let command_buffer = context.command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(input.metal()), 0);
        encoder.set_buffer(1, Some(cos.metal()), 0);
        encoder.set_buffer(2, Some(sin.metal()), 0);
        encoder.set_buffer(3, Some(output.metal()), 0);
        encoder.set_bytes(
            4,
            std::mem::size_of_val(&shape_nd2) as _,
            shape_nd2.as_ptr() as *const _,
        );
        encoder.set_bytes(
            5,
            std::mem::size_of_val(&strides_nd2) as _,
            strides_nd2.as_ptr() as *const _,
        );

        let grid_size =
            MTLSize { width: (shape_nd2[1] / 2) as _, height: shape_nd2[0] as _, depth: 1 };
        let group_size = utils::build_metal_size_with_ones();

        encoder.use_resource(input.metal(), metal::MTLResourceUsage::Read);
        encoder.use_resource(cos.metal(), metal::MTLResourceUsage::Read);
        encoder.use_resource(sin.metal(), metal::MTLResourceUsage::Read);
        encoder.use_resource(output.metal(), metal::MTLResourceUsage::Write);
        encoder.dispatch_thread_groups(grid_size, group_size);
        encoder.end_encoding();
        Ok(output)
    }
}
