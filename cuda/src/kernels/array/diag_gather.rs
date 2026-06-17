use crate::context::{TractCudaStream, cuda_context};
use crate::kernels::launch_args::TractLaunchArgs;
use crate::kernels::{LibraryName, MAX_THREADS, get_cuda_view};
use anyhow::ensure;
use cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg};
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
        ensure!(Self::is_supported_dt(dt), "Unsupported dt {:?} for cuda diag_gather op", dt);
        let tname = DeviceTensor::tname(dt)?;
        Ok(format!("diag_gather_{tname}"))
    }

    pub fn eval(
        &self,
        stream: &TractCudaStream,
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
        stream.synchronize()?;
        Ok(output)
    }

    pub fn dispatch_eval(
        &self,
        stream: &TractCudaStream,
        input: &DeviceTensor,
        offset: i64,
        out_len: usize,
        output: &DeviceTensor,
    ) -> TractResult<()> {
        let rank = input.rank();
        ensure!(rank >= 2);
        ensure!(output.rank() == rank);
        ensure!(output.datum_type() == input.datum_type());
        let in_shape = input.shape();
        let out_shape = output.shape();
        // Output shares all leading axes with input, only the last differs.
        ensure!(in_shape[..rank - 2] == out_shape[..rank - 2]);
        ensure!(in_shape[rank - 2] == out_shape[rank - 2]);
        ensure!(out_shape[rank - 1] == out_len);
        // i64 -> i32 down-cast: model widths are well under i32::MAX in practice.
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

        let i_view = get_cuda_view(input);
        let o_view = get_cuda_view(output);

        let func = cuda_context()
            .load_pipeline(LibraryName::Array, self.kernel_name(input.datum_type())?)?;

        let mut launch_args = TractLaunchArgs::new(stream, &func);
        launch_args.push_view(&i_view);
        launch_args.push_view(&o_view);
        launch_args.push::<i32>(offset_i32);
        launch_args.push::<i32>(batch as i32);
        launch_args.push::<i32>(t_q as i32);
        launch_args.push::<i32>(r_in as i32);
        launch_args.push::<i32>(out_len_i32);
        launch_args.push::<i32>(in_stride_b);
        launch_args.push::<i32>(in_stride_i);
        launch_args.push::<i32>(in_stride_r);
        launch_args.push::<i32>(out_stride_b);
        launch_args.push::<i32>(out_stride_i);
        launch_args.push::<i32>(out_stride_k);

        // Grid: x = out_len cols, y = T_q rows, z = batch.  Threads per block
        // along x = min(out_len, 256) rounded down to multiple of 32.
        let block_x = out_len.clamp(32, MAX_THREADS);
        let grid_x = out_len.div_ceil(block_x);
        let cfg = LaunchConfig {
            grid_dim: (grid_x as _, t_q as _, batch as _),
            block_dim: (block_x as _, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_args.launch(cfg)
    }
}

pub fn cuda_diag_gather_dispatch(
    input: &DeviceTensor,
    offset: i64,
    out_len: usize,
    output: &DeviceTensor,
) -> TractResult<()> {
    crate::with_cuda_stream(|stream| {
        DiagGather.dispatch_eval(stream, input, offset, out_len, output)
    })
}

crate::register_cuda_op!(tract_transformers::ops::diag_gather::DiagGather, |source, node, op| {
    rule_if!(DiagGather::is_supported_dt(source.node_input_facts(node.id)?[0].datum_type));
    Ok(Some(Box::new(tract_gpu::ops::diag_gather::GpuDiagGather::new(
        op.offset.clone(),
        op.out_len.clone(),
        "Cuda",
        cuda_diag_gather_dispatch,
    ))))
});

#[cfg(test)]
mod tests {
    use super::*;
    use tract_core::internal::Tensor;
    use tract_gpu::tensor::IntoDevice;
    use tract_transformers::ops::diag_gather as cpu_dg;

    fn run_against_cpu(shape: &[usize], offset: i64, out_len: usize) -> TractResult<()> {
        use tract_core::plan::TurnState;
        crate::with_cuda_stream(|stream| {
            let len: usize = shape.iter().product();
            let data: Vec<f32> = (0..len).map(|i| i as f32).collect();
            let cpu_in = Tensor::from_shape(shape, &data)?;
            let cuda_in = cpu_in.clone().into_device()?;

            // CPU DiagGather only implements eval_with_session (it resolves
            // TDims against the session's resolved_symbols); pass an empty
            // TurnState since the TDims here are already concrete.
            let cpu_op =
                cpu_dg::DiagGather { offset: (offset as i64).to_dim(), out_len: out_len.to_dim() };
            let session = TurnState::default();
            let cpu_out = cpu_op.eval_with_session(0, &session, tvec![cpu_in.into_tvalue()])?[0]
                .clone()
                .into_tensor();
            let cuda_out = DiagGather.eval(stream, &cuda_in, offset, out_len)?;
            cpu_out
                .close_enough(&cuda_out.to_host()?.into_tensor(), Approximation::Exact)
                .with_context(|| format!("shape={shape:?} offset={offset} out_len={out_len}"))
        })
    }

    #[test]
    fn test_diag_gather_skew_basic() -> TractResult<()> {
        // Classic skew-trick shape: [B*H, T, 2T-1] -> [B*H, T, T] with offset = T-1.
        let t = 4;
        run_against_cpu(&[2, t, 2 * t - 1], (t - 1) as i64, t)
    }

    #[test]
    fn test_diag_gather_rank4_encoder_like() -> TractResult<()> {
        // Mirrors the encoder shape: [B, H, T_q, R].
        let t = 14;
        run_against_cpu(&[1, 8, t, 2 * t - 1], (t - 1) as i64, t)
    }

    #[test]
    fn test_diag_gather_out_of_bounds_zero_fill() -> TractResult<()> {
        // out_len > R so some `r = offset + k - i` fall outside [0, R) and
        // must be zero-filled (matches CPU contract).
        let r = 5;
        let t = 4;
        run_against_cpu(&[1, t, r], 1, 8)
    }

    #[test]
    fn test_diag_gather_partial_overlap() -> TractResult<()> {
        // offset = 0: row i reads input[..., i, -i..-i+out_len], so the
        // first `i` columns of each row are out of bounds and zeroed.
        let t = 4;
        let r = 6;
        run_against_cpu(&[1, t, r], 0, t)
    }

    #[test]
    fn test_diag_gather_rank2() -> TractResult<()> {
        // Smallest valid rank: no leading batch axes.
        run_against_cpu(&[5, 9], 4, 5)
    }
}
