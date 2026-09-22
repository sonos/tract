use crate::encoder::EncoderExt;
use crate::{LibraryName, MetalStream};
use metal::MTLSize;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

tract_core::declare_knob!(
    TRACT_METAL_DISABLE_GDN_CHUNKED,
    bool,
    false,
    "Disable the chunked gated-delta-rule prefill kernel, forcing the \
     per-token threadgroup-parallel kernel for every sequence length."
);

#[allow(clippy::too_many_arguments)]
fn dispatch_eval(
    stream: &MetalStream,
    query: &DeviceTensor,
    key: &DeviceTensor,
    value: &DeviceTensor,
    log_decay: &DeviceTensor,
    beta: &DeviceTensor,
    initial_state: &DeviceTensor,
    output: &DeviceTensor,
    final_state: &DeviceTensor,
    sigmoid_beta: bool,
) -> TractResult<()> {
    ensure!(query.datum_type() == DatumType::F16);
    ensure!(key.datum_type() == DatumType::F16 && value.datum_type() == DatumType::F16);
    ensure!(log_decay.datum_type() == DatumType::F32);
    ensure!(beta.datum_type() == DatumType::F16);
    ensure!(matches!(initial_state.datum_type(), DatumType::F16 | DatumType::F32));
    ensure!(query.shape() == key.shape());
    // Layout matches the CPU op: q/k [b, S, hk, w], v/output [b, S, hv, w]
    // with hv = G * hk (GQA), gates [b, S, hv], state [b, hv, w, w].
    ensure!(query.rank() == 4, "GDN query must be [b, S, hk, w], got {:?}", query.shape());
    let (batch, s_len, k_heads, width) =
        (query.shape()[0], query.shape()[1], query.shape()[2], query.shape()[3]);
    ensure!(
        value.rank() == 4
            && value.shape()[0] == batch
            && value.shape()[1] == s_len
            && value.shape()[3] == width
            && value.shape()[2].is_multiple_of(k_heads),
        "GDN value must be [b, S, G*hk, w] with query/key [b, S, hk, w], got value {:?} vs query {:?}",
        value.shape(),
        query.shape()
    );
    let heads = value.shape()[2];
    ensure!(log_decay.len() == batch * s_len * heads && beta.len() == batch * s_len * heads);
    ensure!(
        initial_state.shape() == [batch, heads, width, width],
        "GDN state must be [b, hv, w, w], got {:?}",
        initial_state.shape()
    );
    ensure!(output.shape() == value.shape() && output.datum_type() == DatumType::F16);
    ensure!(final_state.shape() == initial_state.shape());

    for tensor in [query, key, value, log_decay, beta, initial_state, output, final_state] {
        stream.retain_tensor(tensor);
    }
    let f16_state = initial_state.datum_type() == DatumType::F16;

    // Prefill-shaped steps run the chunked gated delta rule: chunk-local
    // matrices for all chunks in parallel, then a (head, column-block) scan
    // that is sequential only across S/64 chunks instead of S steps.
    // width <= 128: the scan kernel's static threadgroup arrays hold one
    // [width x GDN_COL_BLOCK] f32 state block.
    if s_len >= GDN_CHUNK && width <= 128 && !TRACT_METAL_DISABLE_GDN_CHUNKED.get() {
        return dispatch_chunked(
            stream,
            query,
            key,
            value,
            log_decay,
            beta,
            initial_state,
            output,
            final_state,
            (batch, s_len, k_heads, heads, width),
            f16_state,
            sigmoid_beta,
        );
    }

    // Threadgroup-parallel kernel: one threadgroup per (b, head), threads
    // [width x rchunks], row loops split across chunks and reduced through
    // threadgroup memory. Needs the whole column set in one threadgroup;
    // exotic widths fall back to the thread-per-column kernel.
    let tg_kernel_name =
        if f16_state { "gdn_recurrent_f16_state_f16_tg" } else { "gdn_recurrent_f16_tg" };
    let tg_pipeline = stream.load_pipeline(LibraryName::GdnRecurrent, tg_kernel_name)?;
    let max_tg = tg_pipeline.max_total_threads_per_threadgroup() as usize;
    let mut rchunks = 1;
    while rchunks < 16 && width % (rchunks * 2) == 0 && width * rchunks * 2 <= max_tg.min(1024) {
        rchunks *= 2;
    }

    if width <= max_tg {
        let command_buffer = stream.command_buffer();
        command_buffer.encode(|encoder| {
            encoder.set_compute_pipeline_state(&tg_pipeline);
            for (ix, tensor) in
                [query, key, value, log_decay, beta, initial_state].iter().enumerate()
            {
                encoder.set_metal_tensor(ix as u64, tensor, metal::MTLResourceUsage::Read);
            }
            encoder.set_metal_tensor(6, output, metal::MTLResourceUsage::Write);
            encoder.set_metal_tensor(7, final_state, metal::MTLResourceUsage::Write);
            let n_tgs = (batch * heads) as u64;
            let heads = heads as i32;
            let width_i = width as i32;
            let s_len = s_len as i32;
            let batch_i = batch as i32;
            let k_heads = k_heads as i32;
            encoder.set_bytes(8, size_of::<i32>() as u64, &heads as *const i32 as *const _);
            encoder.set_bytes(9, size_of::<i32>() as u64, &width_i as *const i32 as *const _);
            encoder.set_bytes(10, size_of::<i32>() as u64, &s_len as *const i32 as *const _);
            encoder.set_bytes(11, size_of::<i32>() as u64, &batch_i as *const i32 as *const _);
            encoder.set_bytes(12, size_of::<i32>() as u64, &k_heads as *const i32 as *const _);
            let sigmoid_beta_i = sigmoid_beta as i32;
            encoder.set_bytes(
                13,
                size_of::<i32>() as u64,
                &sigmoid_beta_i as *const i32 as *const _,
            );
            let scratch_bytes = ((2 * rchunks * width + 2 * rchunks) * size_of::<f32>()) as u64;
            encoder.set_threadgroup_memory_length(0, scratch_bytes);
            encoder.dispatch_thread_groups(
                MTLSize { width: n_tgs, height: 1, depth: 1 },
                MTLSize { width: width as u64, height: rchunks as u64, depth: 1 },
            );
        });
        return Ok(());
    }

    let kernel_name = if f16_state { "gdn_recurrent_f16_state_f16" } else { "gdn_recurrent_f16" };
    let pipeline = stream.load_pipeline(LibraryName::GdnRecurrent, kernel_name)?;
    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        for (ix, tensor) in [query, key, value, log_decay, beta, initial_state].iter().enumerate() {
            encoder.set_metal_tensor(ix as u64, tensor, metal::MTLResourceUsage::Read);
        }
        encoder.set_metal_tensor(6, output, metal::MTLResourceUsage::Write);
        encoder.set_metal_tensor(7, final_state, metal::MTLResourceUsage::Write);
        let heads = heads as i32;
        let width = width as i32;
        let s_len = s_len as i32;
        let batch = batch as i32;
        let k_heads = k_heads as i32;
        encoder.set_bytes(8, size_of::<i32>() as u64, &heads as *const i32 as *const _);
        encoder.set_bytes(9, size_of::<i32>() as u64, &width as *const i32 as *const _);
        encoder.set_bytes(10, size_of::<i32>() as u64, &s_len as *const i32 as *const _);
        encoder.set_bytes(11, size_of::<i32>() as u64, &batch as *const i32 as *const _);
        encoder.set_bytes(12, size_of::<i32>() as u64, &k_heads as *const i32 as *const _);
        let sigmoid_beta_i = sigmoid_beta as i32;
        encoder.set_bytes(13, size_of::<i32>() as u64, &sigmoid_beta_i as *const i32 as *const _);
        encoder.dispatch_threads(
            MTLSize { width: (batch * heads * width) as u64, height: 1, depth: 1 },
            MTLSize { width: width.min(1024) as u64, height: 1, depth: 1 },
        );
    });
    Ok(())
}

/// Chunk length of the chunked gated delta rule; must match GDN_CHUNK in
/// gdn_recurrent.metal.
const GDN_CHUNK: usize = 64;
/// State columns per scan threadgroup; must match GDN_COL_BLOCK in
/// gdn_recurrent.metal.
const GDN_COL_BLOCK: usize = 16;

/// Chunked gated delta rule (see gdn_recurrent.metal): one parallel
/// prepare dispatch over (batch, head, chunk), a command-buffer boundary
/// (the scan reads scratch the prepare just wrote: the documented
/// intra-command-buffer write->read defect medicine), then the per-head
/// sequential scan.
#[allow(clippy::too_many_arguments)]
fn dispatch_chunked(
    stream: &MetalStream,
    query: &DeviceTensor,
    key: &DeviceTensor,
    value: &DeviceTensor,
    log_decay: &DeviceTensor,
    beta: &DeviceTensor,
    initial_state: &DeviceTensor,
    output: &DeviceTensor,
    final_state: &DeviceTensor,
    dims: (usize, usize, usize, usize, usize),
    f16_state: bool,
    sigmoid_beta: bool,
) -> TractResult<()> {
    let (batch, s_len, k_heads, heads, width) = dims;
    let nch = s_len.div_ceil(GDN_CHUNK);
    let rows = batch * heads * nch * GDN_CHUNK;
    let f32dt = DatumType::F32;
    let value_p = unsafe { DeviceTensor::uninitialized_dt(f32dt, &[rows, width])? };
    let k_cumdecay = unsafe { DeviceTensor::uninitialized_dt(f32dt, &[rows, width])? };
    let attn_local = unsafe { DeviceTensor::uninitialized_dt(f32dt, &[rows, GDN_CHUNK])? };
    let q_g = unsafe { DeviceTensor::uninitialized_dt(f32dt, &[rows, width])? };
    let w_t = unsafe { DeviceTensor::uninitialized_dt(f32dt, &[rows, width])? };
    let eg_last = unsafe { DeviceTensor::uninitialized_dt(f32dt, &[batch * heads * nch])? };
    for t in [&value_p, &k_cumdecay, &attn_local, &q_g, &w_t, &eg_last] {
        stream.retain_tensor(t);
    }

    let prep = stream.load_pipeline(LibraryName::GdnRecurrent, "gdn_chunk_prepare_f16")?;
    let tg_width = (prep.max_total_threads_per_threadgroup() as u64).min(256);
    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&prep);
        for (ix, t) in [query, key, value, log_decay, beta].iter().enumerate() {
            encoder.set_metal_tensor(ix as u64, t, metal::MTLResourceUsage::Read);
        }
        for (ix, t) in [&value_p, &k_cumdecay, &attn_local, &q_g, &w_t, &eg_last].iter().enumerate()
        {
            encoder.set_metal_tensor(5 + ix as u64, t, metal::MTLResourceUsage::Write);
        }
        encoder.set_slice(11, &[heads as i32]);
        encoder.set_slice(12, &[width as i32]);
        encoder.set_slice(13, &[s_len as i32]);
        encoder.set_slice(14, &[batch as i32]);
        encoder.set_slice(15, &[k_heads as i32]);
        encoder.set_slice(16, &[nch as i32]);
        encoder.set_slice(17, &[sigmoid_beta as i32]);
        encoder.dispatch_thread_groups(
            MTLSize { width: (batch * heads * nch) as u64, height: 1, depth: 1 },
            MTLSize { width: tg_width, height: 1, depth: 1 },
        );
    });
    // Correctness barrier: the scan dispatch below reads the scratch buffers
    // this prepare dispatch just wrote, which requires the prepare command
    // buffer to have actually completed first (see the module doc comment).
    // MetalStream holds one shared, lazily-committed command buffer for the
    // whole stream, so this commits and blocks on EVERY previously-encoded
    // dispatch on the stream, not just this op's own prepare -- a full
    // pipeline drain, not a narrow two-dispatch sync. A non-blocking,
    // per-op-scoped commit needs the stream's buffer-pool/backpressure
    // rework (not part of this PR); until then every chunked-prefill call
    // pays this drain.
    stream.wait_until_completed()?;

    let scan_name = if f16_state { "gdn_chunk_scan_f16_state" } else { "gdn_chunk_scan_f32_state" };
    let scan = stream.load_pipeline(LibraryName::GdnRecurrent, scan_name)?;
    let col_blocks = width.div_ceil(GDN_COL_BLOCK);
    let tg_width = (scan.max_total_threads_per_threadgroup() as u64).min(256);
    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&scan);
        for (ix, t) in [&value_p, &k_cumdecay, &attn_local, &q_g, &w_t, &eg_last].iter().enumerate()
        {
            encoder.set_metal_tensor(ix as u64, t, metal::MTLResourceUsage::Read);
        }
        encoder.set_metal_tensor(6, initial_state, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(7, output, metal::MTLResourceUsage::Write);
        encoder.set_metal_tensor(8, final_state, metal::MTLResourceUsage::Write);
        encoder.set_slice(9, &[heads as i32]);
        encoder.set_slice(10, &[width as i32]);
        encoder.set_slice(11, &[s_len as i32]);
        encoder.set_slice(12, &[batch as i32]);
        encoder.set_slice(13, &[nch as i32]);
        encoder.dispatch_thread_groups(
            MTLSize { width: (batch * heads * col_blocks) as u64, height: 1, depth: 1 },
            MTLSize { width: tg_width, height: 1, depth: 1 },
        );
    });
    // The scratch dies when this function returns. The commit between the
    // two encodes moved the first retains onto the (closed) prepare buffer,
    // which completes long before the scan runs: retain everything again so
    // the scan's command buffer keeps the scratch alive (without this the
    // buffer pool hands the same memory to the next layer's prepare while
    // this scan still reads it: real-model junk output).
    for t in [&value_p, &k_cumdecay, &attn_local, &q_g, &w_t, &eg_last] {
        stream.retain_tensor(t);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn metal_gdn_recurrent_launch(
    query: &DeviceTensor,
    key: &DeviceTensor,
    value: &DeviceTensor,
    log_decay: &DeviceTensor,
    beta: &DeviceTensor,
    initial_state: &DeviceTensor,
    output: &DeviceTensor,
    final_state: &DeviceTensor,
    sigmoid_beta: bool,
) -> TractResult<()> {
    crate::with_metal_stream(|stream| {
        dispatch_eval(
            stream,
            query,
            key,
            value,
            log_decay,
            beta,
            initial_state,
            output,
            final_state,
            sigmoid_beta,
        )
    })
}

crate::register_metal_op!(
    tract_transformers::ops::gdn_recurrent::GatedDeltaNetRecurrent,
    |source, node, op| {
        // The Metal kernel is f16-only (q/k/v/beta f16, gates/state f32);
        // other dtype mixes (e.g. an all-f32 test export) stay on the CPU op.
        let facts = source.node_input_facts(node.id)?;
        let dts: Vec<DatumType> = facts.iter().map(|f| f.datum_type).collect();
        // q/k/v/beta f16, log_decay f32; the recurrent state may be f16
        // (graph exported with -idt f16) or f32, each has its kernel.
        if dts[..5]
            != [DatumType::F16, DatumType::F16, DatumType::F16, DatumType::F32, DatumType::F16]
            || !matches!(dts[5], DatumType::F16 | DatumType::F32)
        {
            return Ok(None);
        }
        Ok(Some(Box::new(tract_gpu::ops::gdn_recurrent::GpuGatedDeltaNetRecurrent {
            backend_name: "Metal",
            dispatch: metal_gdn_recurrent_launch,
            sigmoid_beta: op.sigmoid_beta,
        })))
    }
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::with_borrowed_metal_stream;
    use tract_gpu::tensor::{DeviceTensorExt, IntoDevice};

    #[test]
    fn qwen35_recurrent_step_matches_cpu() -> TractResult<()> {
        with_borrowed_metal_stream(|stream| {
            let heads = 2usize;
            let width = 128usize;
            let vector_len = heads * width;
            let state_len = heads * width * width;
            let qf = (0..vector_len).map(|i| ((i % 31) as f32 - 15.0) / 64.0).collect::<Vec<_>>();
            let kf = (0..vector_len).map(|i| ((i % 29) as f32 - 14.0) / 64.0).collect::<Vec<_>>();
            let vf = (0..vector_len).map(|i| ((i % 23) as f32 - 11.0) / 32.0).collect::<Vec<_>>();
            let sf = (0..state_len).map(|i| ((i % 37) as f32 - 18.0) / 256.0).collect::<Vec<_>>();
            let gf = vec![-0.125f32, -0.75];
            let bf = vec![0.25f32, 0.875];
            let as_f16 = |v: &[f32]| v.iter().copied().map(f16::from_f32).collect::<Vec<_>>();
            let q = Tensor::from_shape(&[1, 1, heads, width], &as_f16(&qf))?.into_device()?;
            let k = Tensor::from_shape(&[1, 1, heads, width], &as_f16(&kf))?.into_device()?;
            let v = Tensor::from_shape(&[1, 1, heads, width], &as_f16(&vf))?.into_device()?;
            let g = Tensor::from_shape(&[1, 1, heads], &gf)?.into_device()?;
            let beta = Tensor::from_shape(&[1, 1, heads], &as_f16(&bf))?.into_device()?;
            let state = Tensor::from_shape(&[1, heads, width, width], &sf)?.into_device()?;
            let output = DeviceTensor::uninitialized_dt(DatumType::F16, q.shape())?;
            let next = DeviceTensor::uninitialized_dt(DatumType::F32, state.shape())?;
            dispatch_eval(stream, &q, &k, &v, &g, &beta, &state, &output, &next, false)?;
            stream.wait_until_completed()?;
            let output = output.to_host()?.into_tensor();
            let next = next.to_host()?.into_tensor();
            let got_output = unsafe { output.as_slice_unchecked::<f16>() };
            let got_state = unsafe { next.as_slice_unchecked::<f32>() };
            for head in 0..heads {
                let base = head * width;
                let matrix = head * width * width;
                let qi =
                    1.0 / (qf[base..base + width].iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt();
                let ki =
                    1.0 / (kf[base..base + width].iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt();
                let decay = gf[head].exp();
                for col in 0..width {
                    let predicted = (0..width)
                        .map(|row| kf[base + row] * ki * sf[matrix + row * width + col] * decay)
                        .sum::<f32>();
                    let residual = (vf[base + col] - predicted) * bf[head];
                    let mut expected_output = 0.0f32;
                    for row in 0..width {
                        let expected_state =
                            sf[matrix + row * width + col] * decay + kf[base + row] * ki * residual;
                        assert!(
                            (got_state[matrix + row * width + col] - expected_state).abs() < 3e-5
                        );
                        expected_output += qf[base + row] * qi * expected_state;
                    }
                    assert_eq!(
                        got_output[base + col],
                        f16::from_f32(expected_output / (width as f32).sqrt())
                    );
                }
            }
            Ok(())
        })
    }

    /// Multi-step Metal-vs-CPU comparison, parametric over the GQA group
    /// count (heads = hv = groups * k_heads) and the state datum type.
    fn multi_step_matches_cpu_op_case(groups: usize, state_dt: DatumType) -> TractResult<()> {
        multi_step_matches_cpu_op_len(groups, state_dt, 5)
    }

    fn multi_step_matches_cpu_op_len(
        groups: usize,
        state_dt: DatumType,
        s_len: usize,
    ) -> TractResult<()> {
        use tract_transformers::ops::gdn_recurrent::GatedDeltaNetRecurrent;
        with_borrowed_metal_stream(|stream| {
            let (b, k_heads, width) = (1usize, 2usize, 32usize);
            let heads = k_heads * groups;
            let n_qk = b * s_len * k_heads * width;
            let n_vec = b * s_len * heads * width;
            let n_gate = b * s_len * heads;
            let n_state = b * heads * width * width;
            let qf = (0..n_qk).map(|i| ((i % 31) as f32 - 15.0) / 64.0).collect::<Vec<_>>();
            let kf = (0..n_qk).map(|i| ((i % 29) as f32 - 14.0) / 64.0).collect::<Vec<_>>();
            let vf = (0..n_vec).map(|i| ((i % 23) as f32 - 11.0) / 32.0).collect::<Vec<_>>();
            let sf = (0..n_state).map(|i| ((i % 37) as f32 - 18.0) / 256.0).collect::<Vec<_>>();
            let gf = (0..n_gate).map(|i| -0.05 - 0.1 * (i % 7) as f32).collect::<Vec<_>>();
            let bf = (0..n_gate).map(|i| 0.125 + 0.08 * (i % 9) as f32).collect::<Vec<_>>();
            let as_f16 = |v: &[f32]| v.iter().copied().map(f16::from_f32).collect::<Vec<_>>();
            let q = Tensor::from_shape(&[b, s_len, k_heads, width], &as_f16(&qf))?;
            let k = Tensor::from_shape(&[b, s_len, k_heads, width], &as_f16(&kf))?;
            let v = Tensor::from_shape(&[b, s_len, heads, width], &as_f16(&vf))?;
            let g = Tensor::from_shape(&[b, s_len, heads], &gf)?;
            let beta = Tensor::from_shape(&[b, s_len, heads], &as_f16(&bf))?;
            let state = Tensor::from_shape(&[b, heads, width, width], &sf)?
                .cast_to_dt(state_dt)?
                .into_owned();

            let cpu = GatedDeltaNetRecurrent::default().eval(
                &EvalContext::out_of_plan(),
                tvec![
                    q.clone().into_tvalue(),
                    k.clone().into_tvalue(),
                    v.clone().into_tvalue(),
                    g.clone().into_tvalue(),
                    beta.clone().into_tvalue(),
                    state.clone().into_tvalue(),
                ],
            )?;

            let qd = q.into_device()?;
            let kd = k.into_device()?;
            let vd = v.into_device()?;
            let gd = g.into_device()?;
            let betad = beta.into_device()?;
            let stated = state.into_device()?;
            let output = DeviceTensor::uninitialized_dt(DatumType::F16, vd.shape())?;
            let next = DeviceTensor::uninitialized_dt(state_dt, stated.shape())?;
            dispatch_eval(stream, &qd, &kd, &vd, &gd, &betad, &stated, &output, &next, false)?;
            stream.wait_until_completed()?;

            let output = output.to_host()?.into_tensor();
            let next = next.to_host()?.into_tensor();
            output
                .cast_to::<f32>()?
                .close_enough(&cpu[0].cast_to::<f32>()?.into_owned(), Approximation::Approximate)?;
            next.cast_to::<f32>()?
                .into_owned()
                .close_enough(&cpu[1].cast_to::<f32>()?.into_owned(), Approximation::Approximate)?;
            Ok(())
        })
    }

    #[test]
    fn qwen35_recurrent_multi_step_matches_cpu_op() -> TractResult<()> {
        multi_step_matches_cpu_op_case(1, DatumType::F32)
    }

    #[test]
    fn qwen35_recurrent_multi_step_grouped_matches_cpu_op() -> TractResult<()> {
        multi_step_matches_cpu_op_case(2, DatumType::F32)
    }

    #[test]
    fn qwen35_recurrent_f16_state_matches_cpu_op() -> TractResult<()> {
        multi_step_matches_cpu_op_case(1, DatumType::F16)
    }

    #[test]
    fn qwen35_recurrent_f16_state_grouped_matches_cpu_op() -> TractResult<()> {
        multi_step_matches_cpu_op_case(2, DatumType::F16)
    }

    /// Prefill-shaped steps (s_len >= 64) take the chunked gated delta rule
    /// path; 200 forces multiple chunks plus a partial last chunk of 8.
    #[test]
    fn qwen35_chunked_prefill_matches_cpu_op() -> TractResult<()> {
        multi_step_matches_cpu_op_len(1, DatumType::F32, 200)
    }

    #[test]
    fn qwen35_chunked_prefill_grouped_matches_cpu_op() -> TractResult<()> {
        multi_step_matches_cpu_op_len(2, DatumType::F32, 200)
    }

    #[test]
    fn qwen35_chunked_prefill_f16_state_matches_cpu_op() -> TractResult<()> {
        multi_step_matches_cpu_op_len(1, DatumType::F16, 200)
    }

    #[test]
    fn qwen35_chunked_prefill_f16_state_grouped_matches_cpu_op() -> TractResult<()> {
        multi_step_matches_cpu_op_len(2, DatumType::F16, 200)
    }

    /// Exactly one full chunk (the threshold boundary).
    #[test]
    fn qwen35_chunked_prefill_single_chunk_matches_cpu_op() -> TractResult<()> {
        multi_step_matches_cpu_op_len(2, DatumType::F32, 64)
    }

    /// Folding the beta sigmoid into the kernel must be BIT-IDENTICAL to the
    /// standalone Metal sigmoid dispatch followed by the unfolded kernel, at
    /// decode (tg kernel) and prefill (chunked) shapes, for both state dtypes.
    fn sigmoid_beta_fold_is_exact_case(state_dt: DatumType, s_len: usize) -> TractResult<()> {
        with_borrowed_metal_stream(|stream| {
            let (b, k_heads, groups, width) = (1usize, 2usize, 2usize, 32usize);
            let heads = k_heads * groups;
            let n_qk = b * s_len * k_heads * width;
            let n_vec = b * s_len * heads * width;
            let n_gate = b * s_len * heads;
            let n_state = b * heads * width * width;
            let as_f16 = |v: Vec<f32>| v.into_iter().map(f16::from_f32).collect::<Vec<_>>();
            let q = Tensor::from_shape(
                &[b, s_len, k_heads, width],
                &as_f16((0..n_qk).map(|i| ((i % 31) as f32 - 15.0) / 64.0).collect()),
            )?
            .into_device()?;
            let k = Tensor::from_shape(
                &[b, s_len, k_heads, width],
                &as_f16((0..n_qk).map(|i| ((i % 29) as f32 - 14.0) / 64.0).collect()),
            )?
            .into_device()?;
            let v = Tensor::from_shape(
                &[b, s_len, heads, width],
                &as_f16((0..n_vec).map(|i| ((i % 23) as f32 - 11.0) / 32.0).collect()),
            )?
            .into_device()?;
            let g = Tensor::from_shape(
                &[b, s_len, heads],
                &(0..n_gate).map(|i| -0.05 - 0.1 * (i % 7) as f32).collect::<Vec<f32>>(),
            )?
            .into_device()?;
            // Raw beta logits, both signs exercised.
            let beta_raw = Tensor::from_shape(
                &[b, s_len, heads],
                &as_f16((0..n_gate).map(|i| ((i % 13) as f32 - 6.0) / 3.0).collect()),
            )?
            .into_device()?;
            let state = Tensor::from_shape(
                &[b, heads, width, width],
                &(0..n_state).map(|i| ((i % 37) as f32 - 18.0) / 256.0).collect::<Vec<f32>>(),
            )?
            .cast_to_dt(state_dt)?
            .into_owned()
            .into_device()?;

            // Reference: standalone Metal sigmoid dispatch, then unfolded GDN.
            let beta_sig = DeviceTensor::uninitialized_dt(DatumType::F16, beta_raw.shape())?;
            crate::kernels::element_wise::dispatch_eval(
                stream,
                &tract_core::ops::nn::Sigmoid {},
                &beta_raw,
                &beta_sig,
            )?;
            let out_ref = DeviceTensor::uninitialized_dt(DatumType::F16, v.shape())?;
            let next_ref = DeviceTensor::uninitialized_dt(state_dt, state.shape())?;
            dispatch_eval(stream, &q, &k, &v, &g, &beta_sig, &state, &out_ref, &next_ref, false)?;

            // Folded: raw beta straight into the kernel.
            let out_fold = DeviceTensor::uninitialized_dt(DatumType::F16, v.shape())?;
            let next_fold = DeviceTensor::uninitialized_dt(state_dt, state.shape())?;
            dispatch_eval(stream, &q, &k, &v, &g, &beta_raw, &state, &out_fold, &next_fold, true)?;
            stream.wait_until_completed()?;

            out_fold
                .to_host()?
                .into_tensor()
                .close_enough(&out_ref.to_host()?.into_tensor(), Approximation::Exact)?;
            next_fold
                .to_host()?
                .into_tensor()
                .close_enough(&next_ref.to_host()?.into_tensor(), Approximation::Exact)?;
            Ok(())
        })
    }

    #[test]
    fn sigmoid_beta_fold_is_exact_decode() -> TractResult<()> {
        sigmoid_beta_fold_is_exact_case(DatumType::F32, 5)
    }

    #[test]
    fn sigmoid_beta_fold_is_exact_decode_f16_state() -> TractResult<()> {
        sigmoid_beta_fold_is_exact_case(DatumType::F16, 5)
    }

    #[test]
    fn sigmoid_beta_fold_is_exact_chunked() -> TractResult<()> {
        sigmoid_beta_fold_is_exact_case(DatumType::F32, 200)
    }

    #[test]
    fn sigmoid_beta_fold_is_exact_chunked_f16_state() -> TractResult<()> {
        sigmoid_beta_fold_is_exact_case(DatumType::F16, 200)
    }

    /// Timing harness on the real qwen3.5-35B GDN geometry (S=512 prefill
    /// chunk). Run explicitly: cargo test -p tract-metal --release
    /// gdn_chunk_bench -- --ignored --nocapture
    #[test]
    #[ignore]
    fn gdn_chunk_bench() -> TractResult<()> {
        with_borrowed_metal_stream(|stream| {
            let (b, s_len, k_heads, groups, width) = (1usize, 512usize, 16usize, 2usize, 128usize);
            let heads = k_heads * groups;
            let n_qk = b * s_len * k_heads * width;
            let n_vec = b * s_len * heads * width;
            let n_gate = b * s_len * heads;
            let n_state = b * heads * width * width;
            let as_f16 = |v: Vec<f32>| v.into_iter().map(f16::from_f32).collect::<Vec<_>>();
            let q = Tensor::from_shape(
                &[b, s_len, k_heads, width],
                &as_f16((0..n_qk).map(|i| ((i % 31) as f32 - 15.0) / 64.0).collect()),
            )?
            .into_device()?;
            let k = Tensor::from_shape(
                &[b, s_len, k_heads, width],
                &as_f16((0..n_qk).map(|i| ((i % 29) as f32 - 14.0) / 64.0).collect()),
            )?
            .into_device()?;
            let v = Tensor::from_shape(
                &[b, s_len, heads, width],
                &as_f16((0..n_vec).map(|i| ((i % 23) as f32 - 11.0) / 32.0).collect()),
            )?
            .into_device()?;
            let g = Tensor::from_shape(
                &[b, s_len, heads],
                &(0..n_gate).map(|i| -0.05 - 0.1 * (i % 7) as f32).collect::<Vec<f32>>(),
            )?
            .into_device()?;
            let beta = Tensor::from_shape(
                &[b, s_len, heads],
                &as_f16((0..n_gate).map(|i| 0.125 + 0.08 * (i % 9) as f32).collect()),
            )?
            .into_device()?;
            let state = Tensor::from_shape(
                &[b, heads, width, width],
                &as_f16((0..n_state).map(|i| ((i % 37) as f32 - 18.0) / 256.0).collect()),
            )?
            .into_device()?;
            let output = DeviceTensor::uninitialized_dt(DatumType::F16, v.shape())?;
            let next = DeviceTensor::uninitialized_dt(DatumType::F16, state.shape())?;
            for _ in 0..5 {
                dispatch_eval(stream, &q, &k, &v, &g, &beta, &state, &output, &next, false)?;
            }
            stream.wait_until_completed()?;
            const N: usize = 50;
            let start = std::time::Instant::now();
            for _ in 0..N {
                dispatch_eval(stream, &q, &k, &v, &g, &beta, &state, &output, &next, false)?;
            }
            stream.wait_until_completed()?;
            let per = start.elapsed().as_secs_f64() / N as f64;
            eprintln!("gdn chunked s=512: {:.1} us/layer-dispatch", per * 1e6);
            TRACT_METAL_DISABLE_GDN_CHUNKED.set(true);
            let start = std::time::Instant::now();
            for _ in 0..N {
                dispatch_eval(stream, &q, &k, &v, &g, &beta, &state, &output, &next, false)?;
            }
            stream.wait_until_completed()?;
            TRACT_METAL_DISABLE_GDN_CHUNKED.clear();
            let per = start.elapsed().as_secs_f64() / N as f64;
            eprintln!("gdn old tg kernel s=512: {:.1} us/layer-dispatch", per * 1e6);
            Ok(())
        })
    }
}
