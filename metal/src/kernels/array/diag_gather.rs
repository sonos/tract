use crate::encoder::EncoderExt;
use crate::{LibraryName, MetalStream};
use anyhow::{Context, ensure};
use metal::MTLSize;
use std::fmt;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DiagGather;

impl fmt::Display for DiagGather {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl DiagGather {
    pub fn is_supported_dt(dt: DatumType) -> bool {
        matches!(dt, DatumType::F32 | DatumType::F16)
    }

    pub fn kernel_name(&self, dt: DatumType) -> TractResult<String> {
        ensure!(Self::is_supported_dt(dt), "Unsupported dt {:?} for metal diag_gather op", dt);
        let tname = DeviceTensor::tname(dt)?;
        Ok(format!("array_ops::diag_gather_{tname}"))
    }

    pub fn eval(
        &self,
        stream: &MetalStream,
        input: &DeviceTensor,
        offset: i64,
        out_len: usize,
    ) -> TractResult<DeviceTensor> {
        let rank = input.rank();
        ensure!(rank >= 2);
        let mut out_shape: TVec<usize> = input.shape().into();
        out_shape[rank - 1] = out_len;
        let output = unsafe { DeviceTensor::uninitialized_dt(input.datum_type(), &out_shape)? };
        self.dispatch_eval(stream, input, offset, out_len, &output)?;
        stream.wait_until_completed()?;
        Ok(output)
    }

    pub fn dispatch_eval(
        &self,
        stream: &MetalStream,
        input: &DeviceTensor,
        offset: i64,
        out_len: usize,
        output: &DeviceTensor,
    ) -> TractResult<()> {
        stream.retain_tensor(input);
        stream.retain_tensor(output);

        let rank = input.rank();
        ensure!(rank >= 2);
        ensure!(output.rank() == rank);
        ensure!(output.datum_type() == input.datum_type());
        let in_shape = input.shape();
        let out_shape = output.shape();
        ensure!(in_shape[..rank - 2] == out_shape[..rank - 2]);
        ensure!(in_shape[rank - 2] == out_shape[rank - 2]);
        ensure!(out_shape[rank - 1] == out_len);

        let offset_i32: i32 = offset.try_into().context("DiagGather offset overflows i32")?;
        let out_len_i32: i32 = out_len.try_into().context("DiagGather out_len overflows i32")?;

        // Flatten the (rank-2) leading axes into one batch axis.  Assumes the
        // leading block is plain row-major (encoder use: rank-4 BxHxTxR with
        // natural strides), so the batch stride is `t_q * (R or out_len)`.
        let in_strides = input.strides();
        let out_strides = output.strides();
        let batch: usize = in_shape[..rank - 2].iter().product();
        let t_q = in_shape[rank - 2];
        let r_in = in_shape[rank - 1];
        let in_stride_b: i32 = if rank >= 3 { (t_q * r_in) as i32 } else { 0 };
        let in_stride_i = in_strides[rank - 2] as i32;
        let in_stride_r = in_strides[rank - 1] as i32;
        let out_stride_b: i32 = if rank >= 3 { (t_q * out_len) as i32 } else { 0 };
        let out_stride_i = out_strides[rank - 2] as i32;
        let out_stride_k = out_strides[rank - 1] as i32;

        let params: [i32; 10] = [
            offset_i32,
            t_q as i32,
            r_in as i32,
            out_len_i32,
            in_stride_b,
            in_stride_i,
            in_stride_r,
            out_stride_b,
            out_stride_i,
            out_stride_k,
        ];

        let pipeline =
            stream.load_pipeline(LibraryName::ArrayOps, &self.kernel_name(input.datum_type())?)?;
        let command_buffer = stream.command_buffer();
        command_buffer.encode(|encoder| {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_metal_tensor(0, input, metal::MTLResourceUsage::Read);
            encoder.set_metal_tensor(1, output, metal::MTLResourceUsage::Write);
            encoder.set_slice(2, &params);
            let grid_size = MTLSize { width: out_len as _, height: t_q as _, depth: batch as _ };
            let group_size = MTLSize { width: 1, height: 1, depth: 1 };
            encoder.dispatch_thread_groups(grid_size, group_size);
        });
        Ok(())
    }
}

pub fn metal_diag_gather_dispatch(
    input: &DeviceTensor,
    offset: i64,
    out_len: usize,
    output: &DeviceTensor,
) -> TractResult<()> {
    crate::with_metal_stream(|stream| {
        DiagGather.dispatch_eval(stream, input, offset, out_len, output)
    })
}

crate::register_metal_op!(tract_transformers::ops::diag_gather::DiagGather, |source, node, op| {
    rule_if!(DiagGather::is_supported_dt(source.node_input_facts(node.id)?[0].datum_type));
    Ok(Some(Box::new(tract_gpu::ops::diag_gather::GpuDiagGather::new(
        op.offset.clone(),
        op.out_len.clone(),
        "Metal",
        metal_diag_gather_dispatch,
    ))))
});

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::with_borrowed_metal_stream;
    use tract_core::internal::Tensor;
    use tract_core::plan::TurnState;
    use tract_gpu::tensor::IntoDevice;
    use tract_transformers::ops::diag_gather as cpu_dg;

    fn run_against_cpu(shape: &[usize], offset: i64, out_len: usize) -> TractResult<()> {
        with_borrowed_metal_stream(|stream| {
            let len: usize = shape.iter().product();
            let data: Vec<f32> = (0..len).map(|i| i as f32).collect();
            let cpu_in = Tensor::from_shape(shape, &data)?;
            let metal_in = cpu_in.clone().into_device()?;

            let cpu_op =
                cpu_dg::DiagGather { offset: (offset as i64).to_dim(), out_len: out_len.to_dim() };
            let session = TurnState::default();
            let cpu_out = cpu_op.eval_with_session(0, &session, tvec![cpu_in.into_tvalue()])?[0]
                .clone()
                .into_tensor();
            let metal_out = DiagGather.eval(stream, &metal_in, offset, out_len)?;
            cpu_out
                .close_enough(&metal_out.to_host()?.into_tensor(), Approximation::Exact)
                .with_context(|| format!("shape={shape:?} offset={offset} out_len={out_len}"))
        })
    }

    #[test]
    fn test_diag_gather_skew_basic() -> TractResult<()> {
        let t = 4;
        run_against_cpu(&[2, t, 2 * t - 1], (t - 1) as i64, t)
    }

    #[test]
    fn test_diag_gather_rank4_encoder_like() -> TractResult<()> {
        let t = 14;
        run_against_cpu(&[1, 8, t, 2 * t - 1], (t - 1) as i64, t)
    }

    #[test]
    fn test_diag_gather_out_of_bounds_zero_fill() -> TractResult<()> {
        let r = 5;
        let t = 4;
        run_against_cpu(&[1, t, r], 1, 8)
    }

    #[test]
    fn test_diag_gather_partial_overlap() -> TractResult<()> {
        let t = 4;
        let r = 6;
        run_against_cpu(&[1, t, r], 0, t)
    }

    #[test]
    fn test_diag_gather_rank2() -> TractResult<()> {
        run_against_cpu(&[5, 9], 4, 5)
    }
}
