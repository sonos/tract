use crate::encoder::EncoderExt;
use crate::kernels::{utils, BroadcastKind};
use crate::MetalTensor;
use crate::{LibraryName, MetalContext};
use anyhow::{ensure, Result};
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
        matches!(dt, DatumType::F32 | DatumType::F16)
    }

    pub fn kernel_name(&self, dt: DatumType, broadcast_kind: BroadcastKind) -> Result<String> {
        ensure!(Self::is_supported_dt(dt), "Unsupport dt {:?} for metal apply rope", dt);
        let tname = MetalTensor::tname(dt)?;
        let broadcast_name = broadcast_kind.to_func_part();
        Ok(format!("nn_ops::apply_rope_{broadcast_name}_{tname}"))
    }

    pub fn eval(
        &self,
        context: &MetalContext,
        input: &MetalTensor,
        cos: &MetalTensor,
        sin: &MetalTensor,
    ) -> Result<MetalTensor> {
        let output = unsafe { MetalTensor::uninitialized_dt(input.datum_type(), input.shape())? };
        self.dispatch_eval(context, input, cos, sin, &output)?;
        context.wait_until_completed()?;
        Ok(output)
    }

    pub fn dispatch_eval(
        &self,
        context: &MetalContext,
        input: &MetalTensor,
        cos: &MetalTensor,
        sin: &MetalTensor,
        output: &MetalTensor,
    ) -> Result<()> {
        ensure!(input.datum_type() == cos.datum_type());
        ensure!(input.datum_type() == sin.datum_type());

        ensure!(cos.shape() == sin.shape());

        input.retain_until_completion();
        output.retain_until_completion();

        ensure!(input.rank() == cos.rank());
        ensure!(input.rank() >= 2 && input.rank() <= 4);

        ensure!(
            input.shape()[input.rank() - 1] % 2 == 0,
            "Rotate half required most inner dimension to be a multiple of 2: {:?}",
            input.shape()
        );

        let cos_sin_strides =
            crate::utils::compute_broadcast_strides::<usize>(cos.shape(), cos.strides())?;

        let broadcast_kind = BroadcastKind::from_rank(input.rank())
            .with_context(|| anyhow!("Unsupported rank for ApplyRope op: {:?}", input.shape(),))?;

        let kernel_name = self.kernel_name(input.datum_type(), broadcast_kind)?;

        let pipeline = context.shared_context().load_pipeline(LibraryName::NNOps, &kernel_name)?;
        let command_buffer = context.command_buffer();
        command_buffer.encode(|encoder| {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_metal_tensor(0, input, metal::MTLResourceUsage::Read);
            encoder.set_metal_tensor(1, cos, metal::MTLResourceUsage::Read);
            encoder.set_metal_tensor(2, sin, metal::MTLResourceUsage::Read);
            encoder.set_metal_tensor(3, output, metal::MTLResourceUsage::Write);
            encoder.set_slice(4, input.shape());
            encoder.set_slice(5, input.strides());
            encoder.set_slice(6, &cos_sin_strides);

            let mut grid_size = utils::build_metal_size_for_shape(input.shape());
            grid_size.width /= 2;

            let group_size = metal::MTLSize { width: 32 as _, height: 32 as _, depth: 1 as _ };
            encoder.dispatch_threads(grid_size, group_size);
            encoder.end_encoding();
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rewrite_rules::BasicApplyRope;
    use crate::IntoMetal;
    use tract_core::internal::Tensor;

    fn run_test_case(shape: &[usize]) -> Result<()> {
        objc::rc::autoreleasepool(|| {
            crate::METAL_CONTEXT.with_borrow(|context| {
                let len = shape.iter().product::<usize>();

                let a = Tensor::from_shape(
                    shape,
                    &(0..len).map(|f| f as f32 / 1000.0).collect::<Vec<_>>(),
                )?;

                let cos = Tensor::from_shape(
                    shape,
                    &(0..len).map(|f| (f as f32).cos()).collect::<Vec<_>>(),
                )?;

                let sin = Tensor::from_shape(
                    shape,
                    &(0..len).map(|f| (f as f32).sin()).collect::<Vec<_>>(),
                )?;

                let metal_a = a.clone().into_metal()?;
                let metal_sin = sin.clone().into_metal()?;
                let metal_cos = cos.clone().into_metal()?;

                let cpu_output = BasicApplyRope.eval(tvec![
                    a.clone().into(),
                    cos.clone().into(),
                    sin.clone().into(),
                ])?[0]
                    .clone()
                    .into_tensor();
                let metal_output = ApplyRope.eval(context, &metal_a, &metal_cos, &metal_sin)?;

                cpu_output
                    .close_enough(&metal_output.to_cpu()?, Approximation::Approximate)
                    .with_context(|| {
                        anyhow!(
                            "Input: {:?} Cpu: {:?}, Metal: {:?}",
                            a.dump(true),
                            cpu_output.dump(true),
                            metal_output.to_cpu().and_then(|it| it.dump(true))
                        )
                    })?;
                Ok(())
            })
        })
    }

    #[test]
    fn test_apply_rope() -> TractResult<()> {
        run_test_case(&[2, 1, 2, 2])?;
        run_test_case(&[2, 4, 4])?;
        run_test_case(&[2, 1, 512, 10])?;
        run_test_case(&[8, 8])?;
        run_test_case(&[1, 10, 512, 24])?;
        run_test_case(&[3, 10, 512, 24])?;
        Ok(())
    }
}
