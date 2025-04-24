use crate::encoder::EncoderExt;
use crate::kernels::utils::compute_broadcast_strides;
use crate::kernels::{utils, BroadcastKind};
use crate::{LibraryName, MetalStream};
use anyhow::ensure;
use std::fmt;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

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

    pub fn kernel_name(&self, dt: DatumType, broadcast_kind: BroadcastKind) -> TractResult<String> {
        ensure!(Self::is_supported_dt(dt), "Unsupport dt {:?} for metal apply rope", dt);
        let tname = DeviceTensor::tname(dt)?;
        let broadcast_name = broadcast_kind.name();
        Ok(format!("nn_ops::apply_rope_{broadcast_name}_{tname}"))
    }

    pub fn eval(
        &self,
        stream: &MetalStream,
        input: &DeviceTensor,
        cos: &DeviceTensor,
        sin: &DeviceTensor,
    ) -> TractResult<DeviceTensor> {
        let output = unsafe { DeviceTensor::uninitialized_dt(input.datum_type(), input.shape())? };
        self.dispatch_eval(stream, input, cos, sin, &output)?;
        stream.wait_until_completed()?;
        Ok(output)
    }

    pub fn dispatch_eval(
        &self,
        stream: &MetalStream,
        input: &DeviceTensor,
        cos: &DeviceTensor,
        sin: &DeviceTensor,
        output: &DeviceTensor,
    ) -> TractResult<()> {
        ensure!(input.datum_type() == cos.datum_type());
        ensure!(input.datum_type() == sin.datum_type());

        ensure!(cos.shape() == sin.shape());

        stream.retain_tensor(input);
        stream.retain_tensor(cos);
        stream.retain_tensor(sin);
        stream.retain_tensor(output);

        ensure!(input.rank() >= 2 && input.rank() <= 4);
        ensure!(cos.rank() <= input.rank());

        let padded_shape = [&tvec![1; input.rank() - cos.rank()], cos.shape()].concat();
        let (padded_cos, padded_sin) =
            (cos.reshaped(padded_shape.clone())?, sin.reshaped(padded_shape)?);

        ensure!(
            input.shape()[input.rank() - 1] % 2 == 0,
            "Rotate half required most inner dimension to be a multiple of 2: {:?}",
            input.shape()
        );

        let cos_sin_strides =
            compute_broadcast_strides::<usize>(padded_cos.shape(), padded_sin.strides())?;

        let broadcast_kind = BroadcastKind::from_rank(input.rank())
            .with_context(|| format!("Unsupported rank for ApplyRope op: {:?}", input.shape(),))?;

        let kernel_name = self.kernel_name(input.datum_type(), broadcast_kind)?;

        let pipeline = stream.load_pipeline(LibraryName::NNOps, &kernel_name)?;
        let command_buffer = stream.command_buffer();
        command_buffer.encode(|encoder| {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_metal_tensor(0, input, metal::MTLResourceUsage::Read);
            encoder.set_metal_tensor(1, &padded_cos, metal::MTLResourceUsage::Read);
            encoder.set_metal_tensor(2, &padded_sin, metal::MTLResourceUsage::Read);
            encoder.set_metal_tensor(3, output, metal::MTLResourceUsage::Write);
            encoder.set_slice(4, input.shape());
            encoder.set_slice(5, input.strides());
            encoder.set_slice(6, &cos_sin_strides);
            encoder.set_slice(7, output.strides());

            let mut grid_size = utils::build_metal_size_for_shape(input.shape());
            grid_size.width /= 2;

            let group_size = metal::MTLSize { width: 32 as _, height: 32 as _, depth: 1 as _ };
            encoder.dispatch_threads(grid_size, group_size);
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::with_borrowed_metal_stream;
    use tract_core::internal::Tensor;
    use tract_gpu::tensor::IntoDevice;
    use tract_transformers::ops::apply_rope;

    fn run_test_case(shape: &[usize]) -> TractResult<()> {
        with_borrowed_metal_stream(|stream| {
            let len = shape.iter().product::<usize>();

            let a = Tensor::from_shape(
                shape,
                &(0..len).map(|f| f as f32 / 1000.0).collect::<Vec<_>>(),
            )?;

            let cos =
                Tensor::from_shape(shape, &(0..len).map(|f| (f as f32).cos()).collect::<Vec<_>>())?;

            let sin =
                Tensor::from_shape(shape, &(0..len).map(|f| (f as f32).sin()).collect::<Vec<_>>())?;

            let metal_a = a.clone().into_device()?;
            let metal_sin = sin.clone().into_device()?;
            let metal_cos = cos.clone().into_device()?;

            let cpu_output = apply_rope::ApplyRope.eval(tvec![
                a.clone().into(),
                cos.clone().into(),
                sin.clone().into(),
            ])?[0]
                .clone()
                .into_tensor();
            let metal_output = ApplyRope.eval(stream, &metal_a, &metal_cos, &metal_sin)?;

            cpu_output
                .close_enough(&metal_output.to_host()?.into_tensor(), Approximation::Approximate)
                .with_context(|| {
                    format!(
                        "Input: {:?} Cpu: {:?}, Metal: {:?}",
                        a.dump(true),
                        cpu_output.dump(true),
                        metal_output.to_host().and_then(|it| it.dump(true))
                    )
                })?;
            Ok(())
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
