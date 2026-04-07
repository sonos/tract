use crate::encoder::EncoderExt;
use crate::{LibraryName, MetalStream};
use metal::{MTLSize, NSUInteger};
use std::ffi::c_void;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LeakyRelu;

impl LeakyRelu {
    pub fn is_supported_dt(dt: DatumType) -> bool {
        matches!(dt, DatumType::F32 | DatumType::F16)
    }

    pub fn kernel_name(&self, dt: DatumType) -> TractResult<String> {
        ensure!(Self::is_supported_dt(dt), "Unsupported dt {:?} for metal leaky_relu", dt);
        let tname = DeviceTensor::tname(dt)?;
        Ok(format!("nn_ops::leaky_relu_{tname}"))
    }

    pub fn dispatch_eval(
        &self,
        stream: &MetalStream,
        input: &DeviceTensor,
        alpha: f32,
        output: &DeviceTensor,
    ) -> TractResult<()> {
        stream.retain_tensor(input);
        stream.retain_tensor(output);

        ensure!(output.shape() == input.shape());
        ensure!(output.datum_type() == input.datum_type());

        let pipeline =
            stream.load_pipeline(LibraryName::NNOps, &self.kernel_name(input.datum_type())?)?;
        let command_buffer = stream.command_buffer();
        command_buffer.encode(|encoder| {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_metal_tensor(0, input, metal::MTLResourceUsage::Read);
            encoder.set_metal_tensor(1, output, metal::MTLResourceUsage::Write);
            encoder.set_bytes(
                2,
                std::mem::size_of::<f32>() as u64,
                &alpha as *const f32 as *const c_void,
            );

            let grid_size = MTLSize { width: output.len() as NSUInteger, height: 1, depth: 1 };
            let group_size = MTLSize { width: 1, height: 1, depth: 1 };
            encoder.dispatch_thread_groups(grid_size, group_size);
        });
        Ok(())
    }
}

pub fn metal_leaky_relu_dispatch(
    alpha: f32,
    input: &DeviceTensor,
    output: &DeviceTensor,
) -> TractResult<()> {
    crate::with_metal_stream(|stream| LeakyRelu.dispatch_eval(stream, input, alpha, output))
}

// LeakyRelu is an ElementWiseMiniOp, so we register under ElementWiseOp's TypeId.
crate::register_metal_op!(tract_core::ops::element_wise::ElementWiseOp, |_source, _node, op| {
    rule_if_some!(leaky = op.0.downcast_ref::<tract_core::ops::nn::LeakyRelu>());
    Ok(Some(Box::new(tract_gpu::ops::leaky_relu::GpuLeakyRelu::new(
        leaky.alpha,
        "Metal",
        metal_leaky_relu_dispatch,
    ))))
});
