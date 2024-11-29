use crate::encoder::EncoderExt;
use crate::kernels::utils;
use crate::MetalTensor;
use crate::{LibraryName, MetalContext};
use anyhow::{ensure, Result};
use metal::MTLSize;
use std::fmt;
use tract_core::internal::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RotateHalf;

impl fmt::Display for RotateHalf {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl RotateHalf {
    pub fn is_supported_dt(dt: DatumType) -> bool {
        matches!(
            dt,
            DatumType::F32
                | DatumType::F16
                | DatumType::I8
                | DatumType::I16
                | DatumType::I32
                | DatumType::I64
        )
    }

    pub fn kernel_name(&self, dt: DatumType) -> Result<String> {
        ensure!(Self::is_supported_dt(dt), "Unsupport dt {:?} for metal rotate half  op", dt);
        let tname = MetalTensor::tname(dt)?;
        Ok(format!("array_ops::rotate_half_nd2_{tname}"))
    }

    pub fn eval(&self, context: &MetalContext, input: &MetalTensor) -> Result<MetalTensor> {
        let output = unsafe { MetalTensor::uninitialized_dt(input.datum_type(), input.shape())? };
        self.dispatch_eval(context, input, &output)?;
        context.wait_until_completed()?;
        Ok(output)
    }

    pub fn dispatch_eval(
        &self,
        context: &MetalContext,
        input: &MetalTensor,
        output: &MetalTensor,
    ) -> Result<()> {
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
        command_buffer.encode(|encoder| {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_metal_tensor(0, input, metal::MTLResourceUsage::Read);
            encoder.set_metal_tensor(1, output, metal::MTLResourceUsage::Write);
            encoder.set_slice(2, &shape_nd2);
            encoder.set_slice(3, &strides_nd2);

            let grid_size =
                MTLSize { width: (shape_nd2[1] / 2) as _, height: shape_nd2[0] as _, depth: 1 };
            let group_size = utils::build_metal_size_with_ones();

            encoder.dispatch_thread_groups(grid_size, group_size);
            encoder.end_encoding();
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rewrite_rules::BasicRotateHalf;
    use crate::IntoMetal;
    use num_traits::AsPrimitive;
    use tract_core::internal::Tensor;

    fn run_test_case<F>(shape: &[usize]) -> Result<()>
    where
        F: Copy + 'static + Datum,
        usize: AsPrimitive<F>,
    {
        objc::rc::autoreleasepool(|| {
            crate::METAL_CONTEXT.with_borrow(|context| {
                let len = shape.iter().product::<usize>();

                let a = Tensor::from_shape(
                    shape,
                    &(0..len).map(|f| -> F { f.as_() }).collect::<Vec<_>>(),
                )?;

                let metal_a = a.clone().into_metal()?;

                let cpu_output =
                    BasicRotateHalf.eval(tvec![a.clone().into()])?[0].clone().into_tensor();
                let metal_output = RotateHalf.eval(context, &metal_a)?;

                cpu_output
                    .close_enough(&metal_output.to_cpu()?, Approximation::Exact)
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
    fn test_rotate_half() -> TractResult<()> {
        run_test_case::<f32>(&[2, 2])?;
        run_test_case::<f32>(&[512, 512])?;
        run_test_case::<f32>(&[10, 8, 8])?;
        run_test_case::<f32>(&[10, 512, 1024])?;
        run_test_case::<f32>(&[10, 512, 1024])?;
        run_test_case::<i8>(&[10, 1, 2])?;
        run_test_case::<i16>(&[10, 256, 4])?;
        run_test_case::<i32>(&[10, 512, 1024])?;
        Ok(())
    }
}
