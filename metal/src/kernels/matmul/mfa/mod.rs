use crate::kernels::matmul::{GemmDispatchParams, GemmKernel};
use crate::{ConstantValues, LibraryName, MetalStream, Value};
use anyhow::ensure;
use metal::{Buffer, MTLSize, NSUInteger};
use std::ffi::c_void;
use std::fmt;
use tract_core::internal::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct MfaGemm;

impl fmt::Display for MfaGemm {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "MfaGemm")
    }
}

impl GemmKernel for MfaGemm {
    fn name() -> &'static str {
        "mfa"
    }

    fn dispatch_eval(
        &self,
        stream: &MetalStream,
        params: GemmDispatchParams,
        a_buffer: &Buffer,
        b_buffer: &Buffer,
        c_buffer: &Buffer,
    ) -> TractResult<()> {
        let GemmDispatchParams {
            dts,
            a_batch,
            m,
            k,
            n,
            transpose_a,
            a_offset,
            transpose_b,
            b_offset,
            c_offset,
            a_strides,
            b_strides,
            ..
        } = params;

        ensure!(
            matches!(dts[0], DatumType::F32 | DatumType::F16),
            "Unsupported datum type for Mfa {:?}",
            dts[0]
        );
        ensure!(
            dts[0] == dts[1] && dts[0] == dts[2],
            "Mfa only supports homogeneous datum types. I: {:?}, {:?}. O: {:?}",
            dts[0],
            dts[1],
            dts[2]
        );

        dispatch_metal_mfa_gemm(
            stream,
            dts[0],
            (a_batch, m, n, k),
            unsafe { std::mem::transmute::<&[isize], &[usize]>(a_strides.as_slice()) },
            a_offset,
            a_buffer,
            transpose_a,
            unsafe { std::mem::transmute::<&[isize], &[usize]>(b_strides.as_slice()) },
            b_offset,
            b_buffer,
            transpose_b,
            c_buffer,
            c_offset,
        )?;

        Ok(())
    }
}

// From https://github.com/huggingface/candle/blob/main/candle-metal-kernels/src/lib.rs
#[allow(clippy::too_many_arguments)]
pub fn dispatch_metal_mfa_gemm(
    stream: &MetalStream,
    dt: DatumType,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs_stride: &[usize],
    lhs_offset: usize,
    lhs_buffer: &Buffer,
    lhs_transpose: bool,
    rhs_stride: &[usize],
    rhs_offset: usize,
    rhs_buffer: &Buffer,
    rhs_transpose: bool,
    output: &Buffer,
    output_offset: usize,
) -> TractResult<()> {
    assert!(rhs_stride.len() >= 2);
    assert!(lhs_stride.len() >= 2);
    let rhs_m1 = rhs_stride[rhs_stride.len() - 1];
    let rhs_m2 = rhs_stride[rhs_stride.len() - 2];
    let lhs_m1 = lhs_stride[lhs_stride.len() - 1];
    let lhs_m2 = lhs_stride[lhs_stride.len() - 2];
    let a_trans = lhs_transpose;
    let b_trans = rhs_transpose;

    if a_trans {
        // (k, m)
        ensure!(
            lhs_m1 == 1 && lhs_m2 == m,
            "Invalid left matmul argument [{lhs_m2}, {lhs_m1}] != [{m}, 1], strides: {:?} {:?} dims: (m: {m}, n: {n}, k: {k})",
            lhs_stride,
            rhs_stride
        );
    } else {
        // (m, k)
        ensure!(
            lhs_m1 == 1 && lhs_m2 == k,
            "Invalid left matmul argument [{lhs_m2}, {lhs_m1}] != [{k}, 1], strides: {:?} {:?} dims: (m: {m}, n: {n}, k: {k})",
            lhs_stride,
            rhs_stride
        );
    }

    if b_trans {
        // (n, k)
        ensure!(
            rhs_m1 == 1 && rhs_m2 == k,
            "Invalid right matmul argument [{rhs_m2}, {rhs_m1}] != [{k}, 1], strides: {:?} {:?} dims: (m: {m}, n: {n}, k: {k})",
            lhs_stride,
            rhs_stride
        );
    } else {
        // (k, n)
        ensure!(
            rhs_m1 == 1 && rhs_m2 == n,
            "Invalid right matmul argument [{rhs_m2}, {rhs_m1}] != [{n}, 1] {:?} {:?} dims: (m: {m}, n: {n}, k: {k})",
            lhs_stride,
            rhs_stride
        );
    }

    let d_trans = false;
    let alpha = 1.0f32;
    let beta = 0.0f32;
    let batched = b > 1;
    let fused_activation = false;
    let fused_bias = false;
    let (m_simd, n_simd, k_simd, m_splits, n_splits) = if m == 1 {
        let m_simd = 8;
        let n_simd = 8;
        let k_simd = 64;
        let m_splits = 1;
        let n_splits = 1;
        (m_simd, n_simd, k_simd, m_splits, n_splits)
    } else {
        let m_simd = 40;
        let n_simd = 40;
        let k_simd = 32;
        let m_splits = 1;
        let n_splits = 1;
        (m_simd, n_simd, k_simd, m_splits, n_splits)
    };
    let constants = Some(ConstantValues::new(vec![
        (0, Value::USize(m)),
        (1, Value::USize(n)),
        (2, Value::USize(k)),
        (10, Value::Bool(a_trans)),
        (11, Value::Bool(b_trans)),
        (13, Value::Bool(d_trans)),
        (20, Value::F32(alpha)),
        (21, Value::F32(beta)),
        (100, Value::Bool(batched)),
        (101, Value::Bool(fused_activation)),
        // Garbage
        (102, Value::Bool(false)),
        (103, Value::Bool(false)),
        (113, Value::Bool(false)),
        (50_000, Value::Bool(false)),
        // End garbage
        (200, Value::U16(m_simd)),
        (201, Value::U16(n_simd)),
        (202, Value::U16(k_simd)),
        (210, Value::U16(m_splits)),
        (211, Value::U16(n_splits)),
        (50_001, Value::Bool(fused_bias)),
    ]));

    let name = match dt {
        DatumType::F32 => "sgemm",
        DatumType::F16 => "hgemm",
        _ => bail!("MFA GEMM only support F32 or F16 tensors"),
    };

    let pipeline = stream.load_pipeline_with_constants(LibraryName::MfaLib, name, constants)?;
    let m_group = m_simd * m_splits;
    let n_group = n_simd * n_splits;

    let a_block_length = m_group * k_simd;
    let b_block_length = k_simd * n_group;

    let mut block_elements = a_block_length + b_block_length;
    if (m % 8 != 0) && (n % 8 != 0) {
        let c_block_length = m_group * n_group;
        block_elements = std::cmp::max(c_block_length, block_elements)
    }
    if fused_bias {
        if d_trans {
            block_elements = std::cmp::max(block_elements, m_group);
        } else {
            block_elements = std::cmp::max(block_elements, n_group);
        }
    }

    let block_bytes = block_elements * dt.size_of() as u16;

    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_threadgroup_memory_length(0, block_bytes.into());
        encoder.set_buffer(0, Some(lhs_buffer), lhs_offset as NSUInteger);
        encoder.set_buffer(1, Some(rhs_buffer), rhs_offset as NSUInteger);
        encoder.set_buffer(2, Some(output), output_offset as NSUInteger);
        // TODO Tensor D

        let grid_z = b;
        if batched {
            let byte_stride_a: usize = lhs_stride[lhs_stride.len() - 3] * dt.size_of();
            let byte_stride_b: usize = rhs_stride[rhs_stride.len() - 3] * dt.size_of();
            let byte_stride_c = m * n * dt.size_of();
            // TODO byte_stride_d
            let byte_stride_d = 0;

            let buffer: Vec<u64> = vec![
                byte_stride_a as _,
                byte_stride_b as _,
                byte_stride_c as _,
                byte_stride_d as _,
            ];
            encoder.set_bytes(
                10,
                (buffer.len() * core::mem::size_of::<u64>()) as NSUInteger,
                buffer.as_ptr() as *const NSUInteger as *const c_void,
            );
        }

        let grid_size = MTLSize {
            width: n.div_ceil(n_group.into()) as NSUInteger,
            height: m.div_ceil(m_group.into()) as NSUInteger,
            depth: grid_z as NSUInteger,
        };
        let group_size =
            MTLSize { width: 32 * (m_splits as u64) * (n_splits as u64), height: 1, depth: 1 };
        encoder.use_resource(lhs_buffer, metal::MTLResourceUsage::Read);
        encoder.use_resource(rhs_buffer, metal::MTLResourceUsage::Read);
        encoder.use_resource(output, metal::MTLResourceUsage::Write);
        encoder.dispatch_thread_groups(grid_size, group_size);
    });
    Ok(())
}

/// Dispatch the vendored MFA fused `attention` kernel (forward pass).
/// Base case: MHA (no GQA), non-causal, no mask, contiguous `[B,H,R,D]` Q/O and
/// `[B,H,C,D]` K/V. ABI reconstructed from the v1.0.1 source + on-GPU reflection:
/// buffers 0=Q,1=K,2=V,3=O; dtype via `Q_data_type`(30)=MTLDataType raw value;
/// grid=(ceil(R/(R_simd*R_splits)),H,B), group=(32*R_splits,1,1).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_metal_mfa_attention(
    stream: &MetalStream,
    dt: DatumType,
    (b, h, r, c, d): (usize, usize, usize, usize, usize),
    scale: f32,
    causal: bool,
    mask: Option<&Buffer>,
    q_buffer: &Buffer,
    q_offset: usize,
    k_buffer: &Buffer,
    k_offset: usize,
    v_buffer: &Buffer,
    v_offset: usize,
    o_buffer: &Buffer,
    o_offset: usize,
) -> TractResult<()> {
    ensure!(matches!(dt, DatumType::F32 | DatumType::F16), "MFA attention: F32/F16 only");
    // Uniform tiling (R_simd=8, C_simd=32, R_splits=4) for all dtypes — MFA v1.0.1's host
    // f32 defaults. `fuse_async_loads` stays off: the async double-buffer it drives needs more
    // threadgroup memory than we currently size, and enabling it regressed f16 precision ~1000x
    // (0.00003 -> 0.034). MFA also publishes an f16 head-dim SIMD table, but it only pays off
    // with the async path on, so the uniform params are at parity here and simpler.
    let (r_simd, c_simd, r_splits): (u16, u16, u16) = (8, 32, 4);
    let fuse_async = false;
    let r_group = (r_simd as usize) * (r_splits as usize);
    let q_data_type = match dt {
        DatumType::F32 => metal::MTLDataType::Float as usize,
        DatumType::F16 => metal::MTLDataType::Half as usize,
        _ => unreachable!(),
    };
    let batched = b > 1;
    let constants = Some(ConstantValues::new(vec![
        (0, Value::USize(r)),     // R rows (query len)
        (1, Value::USize(c)),     // C cols (key len)
        (2, Value::USize(h)),     // H heads
        (3, Value::USize(d)),     // D head dim
        (10, Value::Bool(false)), // Q_trans
        (11, Value::Bool(false)), // K_trans
        (12, Value::Bool(false)), // V_trans
        (13, Value::Bool(false)), // O_trans
        (20, Value::F32(scale)),  // alpha
        (30, Value::USize(q_data_type)),
        (100, Value::Bool(batched)),              // batched
        (102, Value::Bool(false)),                // block_sparse
        (103, Value::Bool(causal)),               // triangular (causal)
        (110, Value::Bool(true)),                 // forward
        (111, Value::Bool(false)),                // backward
        (112, Value::Bool(false)),                // generate_block_mask
        (113, Value::Bool(false)),                // grouped_query
        (114, Value::Bool(dt == DatumType::F16)), // float_accumulator
        (200, Value::U16(r_simd)),
        (201, Value::U16(c_simd)),
        (210, Value::U16(r_splits)),
        (213, Value::Bool(fuse_async)),        // fuse_async_loads
        (220, Value::U16(0)),                  // R_bank_offset
        (221, Value::U16(0)),                  // C_bank_offset
        (222, Value::U16(0)),                  // D_bank_offset
        (50_000, Value::Bool(mask.is_some())), // masked
    ]));

    let pipeline =
        stream.load_pipeline_with_constants(LibraryName::MfaLib, "attention", constants)?;

    // threadgroup block: max tile (R_group or C_simd) x D, generous within the 32 KiB limit.
    let block_elements = std::cmp::max(r_group, c_simd as usize) * (d + 8);
    let block_bytes = (block_elements * dt.size_of()).min(32 * 1024);

    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_threadgroup_memory_length(0, block_bytes as NSUInteger);
        encoder.set_buffer(0, Some(q_buffer), q_offset as NSUInteger);
        encoder.set_buffer(1, Some(k_buffer), k_offset as NSUInteger);
        encoder.set_buffer(2, Some(v_buffer), v_offset as NSUInteger);
        encoder.set_buffer(3, Some(o_buffer), o_offset as NSUInteger);
        if let Some(m) = mask {
            encoder.set_buffer(12, Some(m), 0);
        }

        let grid_size = MTLSize {
            width: r.div_ceil(r_group) as NSUInteger,
            height: h as NSUInteger,
            depth: b as NSUInteger,
        };
        let group_size = MTLSize { width: 32 * r_splits as NSUInteger, height: 1, depth: 1 };
        encoder.use_resource(q_buffer, metal::MTLResourceUsage::Read);
        encoder.use_resource(k_buffer, metal::MTLResourceUsage::Read);
        encoder.use_resource(v_buffer, metal::MTLResourceUsage::Read);
        encoder.use_resource(o_buffer, metal::MTLResourceUsage::Write);
        if let Some(m) = mask {
            encoder.use_resource(m, metal::MTLResourceUsage::Read);
        }
        encoder.dispatch_thread_groups(grid_size, group_size);
    });
    Ok(())
}

/// Adapter: run fused MFA attention on tract's native head-major layout
/// (Q/K/V = `[H,S,D]`, batch 1), permuting into MFA's layout (Q=[R,H,D],
/// K=[H,D,C], V=[C,H,D]) on-device via `copy_nd`, and the output back to
/// `[H,R,D]`. The K transpose is the one unavoidable copy (candidate to fold,
/// idea #3); Q/V/O permutes are head-major↔head-minor reorders.
#[allow(clippy::too_many_arguments)]
pub fn mfa_attention_head_major(
    stream: &MetalStream,
    dt: DatumType,
    scale: f32,
    causal: bool,
    mask: Option<&tract_gpu::tensor::DeviceTensor>, // additive [R,C], broadcast over heads
    q: &tract_gpu::tensor::DeviceTensor,            // contiguous, logical [H, R, D]
    k: &tract_gpu::tensor::DeviceTensor,            // contiguous, logical [H, C, D]
    v: &tract_gpu::tensor::DeviceTensor,            // contiguous, logical [H, C, D]
    (h, r, c, d): (usize, usize, usize, usize),
    out: &tract_gpu::tensor::DeviceTensor, // contiguous, logical [H, R, D]
) -> TractResult<()> {
    use tract_gpu::tensor::DeviceTensor;
    // out axis i reads input axis `pick[i]`; output is contiguous.
    let permute = |input: &DeviceTensor,
                   in_shape: &[usize],
                   out_shape: &[usize],
                   pick: &[usize]|
     -> TractResult<DeviceTensor> {
        let in_nat = Tensor::natural_strides(in_shape);
        let in_strides: Vec<isize> = pick.iter().map(|&a| in_nat[a]).collect();
        let o = unsafe { DeviceTensor::uninitialized_dt(dt, out_shape)? };
        let out_strides = Tensor::natural_strides(out_shape);
        crate::kernels::array::metal_copy_nd_dispatch(
            input,
            0,
            &in_strides,
            &o,
            0,
            out_shape,
            &out_strides,
        )?;
        Ok(o)
    };
    let qn = permute(q, &[h, r, d], &[r, h, d], &[1, 0, 2])?; // [R,H,D]
    let kn = permute(k, &[h, c, d], &[h, d, c], &[0, 2, 1])?; // [H,D,C]
    let vn = permute(v, &[h, c, d], &[c, h, d], &[1, 0, 2])?; // [C,H,D]
    let on = unsafe { DeviceTensor::uninitialized_dt(dt, &[r, h, d])? };
    // keep intermediates + mask alive until the command buffer completes
    stream.retain_tensor(&qn);
    stream.retain_tensor(&kn);
    stream.retain_tensor(&vn);
    stream.retain_tensor(&on);
    stream.retain_tensor(out);
    if let Some(m) = mask {
        stream.retain_tensor(m);
    }
    dispatch_metal_mfa_attention(
        stream,
        dt,
        (1, h, r, c, d),
        scale,
        causal,
        mask.map(crate::utils::get_metal_buffer),
        crate::utils::get_metal_buffer(&qn),
        0,
        crate::utils::get_metal_buffer(&kn),
        0,
        crate::utils::get_metal_buffer(&vn),
        0,
        crate::utils::get_metal_buffer(&on),
        0,
    )?;
    // un-permute on[R,H,D] -> out[H,R,D] (out buffer is contiguous [H,R,D])
    let on_nat = Tensor::natural_strides(&[r, h, d]);
    let in_strides: Vec<isize> = [1usize, 0, 2].iter().map(|&a| on_nat[a]).collect();
    let out_strides = Tensor::natural_strides(&[h, r, d]);
    crate::kernels::array::metal_copy_nd_dispatch(
        &on,
        0,
        &in_strides,
        out,
        0,
        &[h, r, d],
        &out_strides,
    )?;
    Ok(())
}

/// Metal device op: fused SDPA over `[B,H,Sq,D]` Q / `[B,H,Sk,D]` K,V via the
/// vendored MFA kernel. `B>1` is folded to `B*H` heads; causal is realized as a
/// `[Sq,Sk]` additive mask (the `triangular` constant is a no-op).
#[derive(Debug, Clone)]
pub struct MetalMfaSdpa {
    pub scale: f32,
    pub is_causal: bool,
}

impl PartialEq for MetalMfaSdpa {
    fn eq(&self, o: &Self) -> bool {
        self.scale.to_bits() == o.scale.to_bits() && self.is_causal == o.is_causal
    }
}
impl Eq for MetalMfaSdpa {}
impl std::hash::Hash for MetalMfaSdpa {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.scale.to_bits().hash(state);
        self.is_causal.hash(state);
    }
}

impl Op for MetalMfaSdpa {
    fn name(&self) -> StaticName {
        "MetalMfaSdpa".into()
    }
    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("scale={} causal={}", self.scale, self.is_causal)])
    }
    op_as_typed_op!();
}

impl EvalOp for MetalMfaSdpa {
    fn is_stateless(&self) -> bool {
        true
    }
    fn eval_with_session(
        &self,
        node_id: usize,
        session: &TurnState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        use tract_gpu::tensor::{DeviceTensorExt, IntoDevice};
        ensure!(inputs.len() == 3, "MetalMfaSdpa expects Q,K,V");
        let q = inputs[0].to_device_tensor()?;
        let k = inputs[1].to_device_tensor()?;
        let v = inputs[2].to_device_tensor()?;
        let qs = q.shape();
        ensure!(qs.len() == 4, "expects rank-4 [B,H,Sq,D], got {qs:?}");
        let (b, nh, sq, dd) = (qs[0], qs[1], qs[2], qs[3]);
        ensure!(
            k.shape()[1] == nh && v.shape()[1] == nh,
            "MFA attention needs equal Q/K/V head counts, got Q={nh} K={} V={}",
            k.shape()[1],
            v.shape()[1]
        );
        let sk = k.shape()[2];
        let h = b * nh;
        let dt = q.datum_type();
        let out = tract_gpu::session_handler::make_tensor_for_node(
            session,
            node_id,
            dt,
            &[b, nh, sq, dd],
        )?;
        // causal -> [Sq,Sk] additive mask, bottom-right aligned (cached/cross attention)
        let mask = if self.is_causal {
            let mut m = vec![0f32; sq * sk];
            for i in 0..sq {
                for j in 0..sk {
                    if j as isize > i as isize + (sk as isize - sq as isize) {
                        m[i * sk + j] = -1e30;
                    }
                }
            }
            Some(Tensor::from_shape(&[sq, sk], &m)?.cast_to_dt(dt)?.into_owned().into_device()?)
        } else {
            None
        };
        crate::with_metal_stream(|stream| {
            mfa_attention_head_major(
                stream,
                dt,
                self.scale,
                false,
                mask.as_ref(),
                q,
                k,
                v,
                (h, sq, sk, dd),
                &out,
            )
        })?;
        Ok(tvec![out.into_tensor().into_tvalue()])
    }
}

impl TypedOp for MetalMfaSdpa {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        tract_gpu::utils::facts_to_device_facts(inputs, |f| Ok(tvec![f[0].without_value()]))
    }
    as_op!();
}

/// Whether an `Sdpa` node can be fused by the MFA kernel: exactly Q,K,V (causal
/// or no external mask), f16/f32, rank-4 `[B,H,S,D]`, equal & concrete Q/K/V
/// head count (the 2023 kernel predates GQA), equal & concrete Q/V head dim
/// (multiple of 8, ≤256). Unsupported → `None` → explode/CPU fallback.
pub fn mfa_sdpa_supported(
    op: &tract_transformers::ops::sdpa::Sdpa,
    in_facts: &[&TypedFact],
) -> bool {
    if in_facts.len() != 3 {
        return false; // 4th input = external mask: not wired yet
    }
    let (q, k, v) = (in_facts[0], in_facts[1], in_facts[2]);
    if !matches!(q.datum_type, DatumType::F16 | DatumType::F32) || !op.acc_datum_type.is_float() {
        return false;
    }
    if q.rank() != 4 || k.rank() != 4 || v.rank() != 4 {
        return false;
    }
    match (q.shape[1].to_usize().ok(), k.shape[1].to_usize().ok(), v.shape[1].to_usize().ok()) {
        (Some(qh), Some(kh), Some(vh)) if qh == kh && kh == vh => {}
        _ => return false, // GQA (H_kv < H_q) or symbolic head count
    }
    match (q.shape[3].to_usize().ok(), v.shape[3].to_usize().ok()) {
        (Some(qd), Some(vd)) => qd == vd && qd % 8 == 0 && qd <= 256,
        _ => false,
    }
}

crate::register_metal_op!(tract_transformers::ops::sdpa::Sdpa, |source, node, op| {
    let in_facts = source.node_input_facts(node.id)?;
    if !mfa_sdpa_supported(op, &in_facts) {
        return Ok(None);
    }
    let head_dim = in_facts[0].shape[in_facts[0].rank() - 1].to_usize()?;
    let scale = match &op.scale {
        Some(t) => t.cast_to_scalar::<f32>()?,
        None => (head_dim as f32).recip().sqrt(),
    };
    Ok(Some(Box::new(MetalMfaSdpa { scale, is_causal: op.is_causal }) as Box<dyn TypedOp>))
});

#[cfg(test)]
mod tests {
    use crate::utils::with_borrowed_metal_stream;

    use super::*;
    use crate::kernels::matmul::GemmImpl;
    use tract_gpu::tensor::{DeviceTensor, IntoDevice};

    #[test]
    fn test_mfa_gemm() -> TractResult<()> {
        with_borrowed_metal_stream(|stream| {
            let (b, m, n, k) = (1, 2, 4, 3);
            let a = Tensor::from_shape(
                &[b, m, k],
                &(0..b * m * k).map(|f| f as f32).collect::<Vec<_>>(),
            )?
            .into_device()?;
            let b = Tensor::from_shape(
                &[b, k, n],
                &(0..b * n * k).map(|f| f as f32).collect::<Vec<_>>(),
            )?
            .into_device()?;

            let c = GemmImpl::<MfaGemm>::default().eval(stream, &a, &b)?;

            let expected_c =
                Tensor::from_shape(&[1, 2, 4], &[20.0, 23.0, 26.0, 29.0, 56.0, 68.0, 80.0, 92.0])?;

            let c = c.to_host()?;
            assert!(c.close_enough(&expected_c, Approximation::Close).is_ok());

            let (b, m, n, k) = (2, 2, 4, 3);
            let a = DeviceTensor::from_shape(
                &[b, m, k],
                &(0..b * m * k).map(|f| f as f32).collect::<Vec<_>>(),
            )?;
            let b = DeviceTensor::from_shape(
                &[b, k, n],
                &(0..b * n * k).map(|f| f as f32).collect::<Vec<_>>(),
            )?;

            let c = GemmImpl::<MfaGemm>::default().eval(stream, &a, &b)?;

            let expected_c = Tensor::from_shape(
                &[2, 2, 4],
                &[
                    20.0, 23.0, 26.0, 29.0, 56.0, 68.0, 80.0, 92.0, 344.0, 365.0, 386.0, 407.0,
                    488.0, 518.0, 548.0, 578.0,
                ],
            )?;

            assert!(c.to_host()?.close_enough(&expected_c, Approximation::Close).is_ok());
            Ok(())
        })
    }

    // PR #2 base case: fused MFA attention vs an independent host softmax(QK^T)V.
    #[test]
    fn test_mfa_attention_f32() -> TractResult<()> {
        use crate::utils::get_metal_buffer;
        use tract_gpu::tensor::{DeviceTensor, IntoDevice};
        with_borrowed_metal_stream(|stream| {
            // MFA layout contract (v1.0.1): Q/O = [R,H,D] (head-minor), K = [H,D,C]
            // (K stored transposed), V = [C,H,D]; all trans flags false.
            let rng = |n: usize, seed: u64| -> Vec<f32> {
                let mut s = seed;
                (0..n)
                    .map(|_| {
                        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
                    })
                    .collect()
            };
            // one case in MFA native layout; returns max_abs vs host softmax(QK^T)V
            let run = |h: usize, r: usize, c: usize, d: usize| -> TractResult<f32> {
                let scale = 1.0f32 / (d as f32).sqrt();
                let qv = rng(r * h * d, h as u64 * 7 + 1); // Q [R,H,D]
                let kv = rng(h * d * c, h as u64 * 7 + 2); // K [H,D,C]
                let vv = rng(c * h * d, h as u64 * 7 + 3); // V [C,H,D]
                let mut want = vec![0f32; r * h * d];
                for hh in 0..h {
                    for i in 0..r {
                        let mut sc = vec![0f32; c];
                        for j in 0..c {
                            let mut acc = 0f32;
                            for dd in 0..d {
                                acc += qv[(i * h + hh) * d + dd] * kv[(hh * d + dd) * c + j];
                            }
                            sc[j] = acc * scale;
                        }
                        let m = sc.iter().copied().fold(f32::MIN, f32::max);
                        let mut sum = 0f32;
                        for x in sc.iter_mut() {
                            *x = (*x - m).exp();
                            sum += *x;
                        }
                        for x in sc.iter_mut() {
                            *x /= sum;
                        }
                        for e in 0..d {
                            let mut acc = 0f32;
                            for j in 0..c {
                                acc += sc[j] * vv[(j * h + hh) * d + e];
                            }
                            want[(i * h + hh) * d + e] = acc;
                        }
                    }
                }
                let qd = Tensor::from_shape(&[r, h, d], &qv)?.into_device()?;
                let kd = Tensor::from_shape(&[h, d, c], &kv)?.into_device()?;
                let vd = Tensor::from_shape(&[c, h, d], &vv)?.into_device()?;
                let od = unsafe { DeviceTensor::uninitialized_dt(DatumType::F32, &[r, h, d])? };
                dispatch_metal_mfa_attention(
                    stream,
                    DatumType::F32,
                    (1, h, r, c, d),
                    scale,
                    false,
                    None,
                    get_metal_buffer(&qd),
                    0,
                    get_metal_buffer(&kd),
                    0,
                    get_metal_buffer(&vd),
                    0,
                    get_metal_buffer(&od),
                    0,
                )?;
                let got = od.to_host()?.into_tensor();
                let gv = unsafe { got.as_slice_unchecked::<f32>() };
                let max_abs =
                    gv.iter().zip(want.iter()).map(|(&g, &w)| (g - w).abs()).fold(0f32, f32::max);
                Ok(max_abs)
            };

            for (h, r, c, d) in [(1, 64, 64, 64), (2, 64, 64, 64), (4, 32, 48, 64), (3, 40, 40, 32)]
            {
                let e = run(h, r, c, d)?;
                println!("  H={h} R={r} C={c} D={d}: max_abs={e:.6}");
                ensure!(e < 1e-3, "MFA attention mismatch H={h}: max_abs={e}");
            }
            println!("MFA attention f32: all cases match host reference ✓");
            Ok(())
        })
    }

    // f16 path: same kernel with Q_data_type=Half + float_accumulator, vs an f32
    // reference over f16-rounded inputs (so only the kernel's f16 compute is judged).
    #[test]
    fn test_mfa_attention_f16() -> TractResult<()> {
        use crate::utils::get_metal_buffer;
        use tract_gpu::tensor::{DeviceTensor, IntoDevice};
        with_borrowed_metal_stream(|stream| {
            let (h, r, c, d) = (1usize, 64usize, 64usize, 64usize);
            let scale = 1.0f32 / (d as f32).sqrt();
            let rng = |n: usize, seed: u64| -> Vec<f32> {
                let mut s = seed;
                (0..n)
                    .map(|_| {
                        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
                    })
                    .collect()
            };
            let to16 = |v: Vec<f32>| -> Vec<f32> {
                v.into_iter().map(|x| f16::from_f32(x).to_f32()).collect()
            };
            let qv = to16(rng(r * d, 1)); // Q [R,D]
            let kdc = to16(rng(d * c, 2)); // K [D,C]
            let vv = to16(rng(c * d, 3)); // V [C,D]
            let mut want = vec![0f32; r * d];
            for i in 0..r {
                let mut sc = vec![0f32; c];
                for j in 0..c {
                    let mut a = 0f32;
                    for dd in 0..d {
                        a += qv[i * d + dd] * kdc[dd * c + j];
                    }
                    sc[j] = a * scale;
                }
                let m = sc.iter().copied().fold(f32::MIN, f32::max);
                let mut sum = 0f32;
                for x in sc.iter_mut() {
                    *x = (*x - m).exp();
                    sum += *x;
                }
                for x in sc.iter_mut() {
                    *x /= sum;
                }
                for e in 0..d {
                    let mut a = 0f32;
                    for j in 0..c {
                        a += sc[j] * vv[j * d + e];
                    }
                    want[i * d + e] = a;
                }
            }
            let mk = |shape: &[usize], data: &[f32]| -> TractResult<DeviceTensor> {
                Tensor::from_shape(shape, data)?.cast_to::<f16>()?.into_owned().into_device()
            };
            let q = mk(&[r, d], &qv)?;
            let k = mk(&[d, c], &kdc)?;
            let v = mk(&[c, d], &vv)?;
            let o = unsafe { DeviceTensor::uninitialized_dt(DatumType::F16, &[r, d])? };
            dispatch_metal_mfa_attention(
                stream,
                DatumType::F16,
                (1, h, r, c, d),
                scale,
                false,
                None,
                get_metal_buffer(&q),
                0,
                get_metal_buffer(&k),
                0,
                get_metal_buffer(&v),
                0,
                get_metal_buffer(&o),
                0,
            )?;
            let got = o.to_host()?.cast_to::<f32>()?.into_owned();
            let gv = unsafe { got.as_slice_unchecked::<f32>() };
            let max_abs =
                gv.iter().zip(want.iter()).map(|(&g, &w)| (g - w).abs()).fold(0f32, f32::max);
            println!("  f16 attention: max_abs={max_abs:.5} (f16 tolerance)");
            ensure!(max_abs < 5e-2, "f16 mismatch: {max_abs}");
            println!("MFA attention f16 (H=1): matches f32 reference within f16 tolerance ✓");
            Ok(())
        })
    }

    // f16 correctness across head dims (uniform tiling), vs an f32 reference over
    // f16-rounded inputs.
    #[test]
    fn test_mfa_attention_f16_head_dims() -> TractResult<()> {
        use crate::utils::get_metal_buffer;
        use tract_gpu::tensor::{DeviceTensor, IntoDevice};
        with_borrowed_metal_stream(|stream| {
            let (r, c) = (64usize, 64usize);
            let rng = |n: usize, seed: u64| -> Vec<f32> {
                let mut st = seed;
                (0..n)
                    .map(|_| {
                        st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        ((st >> 40) as f32 / (1u64 << 24) as f32) - 0.5
                    })
                    .collect()
            };
            let to16 = |v: Vec<f32>| -> Vec<f32> {
                v.into_iter().map(|x| f16::from_f32(x).to_f32()).collect()
            };
            // covers branches: 16->C72, 24->C56, 56->C64, 64->C40, 96->C64, 128->C32
            for d in [16usize, 24, 56, 64, 96, 128] {
                let scale = 1.0f32 / (d as f32).sqrt();
                let qv = to16(rng(r * d, 1));
                let kdc = to16(rng(d * c, 2));
                let vv = to16(rng(c * d, 3));
                let mut want = vec![0f32; r * d];
                for i in 0..r {
                    let mut sc = vec![0f32; c];
                    for j in 0..c {
                        let mut a = 0f32;
                        for dd in 0..d {
                            a += qv[i * d + dd] * kdc[dd * c + j];
                        }
                        sc[j] = a * scale;
                    }
                    let m = sc.iter().copied().fold(f32::MIN, f32::max);
                    let mut sum = 0f32;
                    for x in sc.iter_mut() {
                        *x = (*x - m).exp();
                        sum += *x;
                    }
                    for x in sc.iter_mut() {
                        *x /= sum;
                    }
                    for e in 0..d {
                        let mut a = 0f32;
                        for j in 0..c {
                            a += sc[j] * vv[j * d + e];
                        }
                        want[i * d + e] = a;
                    }
                }
                let mk = |sh: &[usize], data: &[f32]| -> TractResult<DeviceTensor> {
                    Tensor::from_shape(sh, data)?.cast_to::<f16>()?.into_owned().into_device()
                };
                let q = mk(&[r, d], &qv)?;
                let k = mk(&[d, c], &kdc)?;
                let v = mk(&[c, d], &vv)?;
                let o = unsafe { DeviceTensor::uninitialized_dt(DatumType::F16, &[r, d])? };
                dispatch_metal_mfa_attention(
                    stream,
                    DatumType::F16,
                    (1, 1, r, c, d),
                    scale,
                    false,
                    None,
                    get_metal_buffer(&q),
                    0,
                    get_metal_buffer(&k),
                    0,
                    get_metal_buffer(&v),
                    0,
                    get_metal_buffer(&o),
                    0,
                )?;
                let got = o.to_host()?.cast_to::<f32>()?.into_owned();
                let gv = unsafe { got.as_slice_unchecked::<f32>() };
                let max_abs =
                    gv.iter().zip(want.iter()).map(|(&g, &w)| (g - w).abs()).fold(0f32, f32::max);
                println!("  f16 D={d}: max_abs={max_abs:.5}");
                ensure!(max_abs < 5e-3, "f16 D={d} mismatch: max_abs={max_abs}");
            }
            println!("MFA f16: correct across head dims ✓");
            Ok(())
        })
    }

    // FINDING (documented): the `triangular` FUNCTION-CONSTANT alone does NOT mask —
    // output == full attention. MFA v1.0.1 causal needs the `triangular_pass` runtime
    // arg + a block-mask pre-pass. Practical route for tract = causal via additive mask
    // (the `masked` path, buffer 12) — see test_mfa_attention_masked.
    #[test]
    fn test_mfa_attention_causal_const_is_noop() -> TractResult<()> {
        use crate::utils::get_metal_buffer;
        use tract_gpu::tensor::{DeviceTensor, IntoDevice};
        with_borrowed_metal_stream(|stream| {
            let (h, s, d) = (1usize, 64usize, 64usize);
            let (r, c) = (s, s);
            let scale = 1.0f32 / (d as f32).sqrt();
            let rng = |n: usize, seed: u64| -> Vec<f32> {
                let mut st = seed;
                (0..n)
                    .map(|_| {
                        st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        ((st >> 40) as f32 / (1u64 << 24) as f32) - 0.5
                    })
                    .collect()
            };
            let qv = rng(r * d, 1);
            let kdc = rng(d * c, 2);
            let vv = rng(c * d, 3);
            // reference attention keeping keys j where keep(i,j) is true
            let reference = |keep: &dyn Fn(usize, usize) -> bool| -> Vec<f32> {
                let mut want = vec![0f32; r * d];
                for i in 0..r {
                    let js: Vec<usize> = (0..c).filter(|&j| keep(i, j)).collect();
                    let mut sc = vec![0f32; js.len()];
                    for (jx, &j) in js.iter().enumerate() {
                        let mut a = 0f32;
                        for dd in 0..d {
                            a += qv[i * d + dd] * kdc[dd * c + j];
                        }
                        sc[jx] = a * scale;
                    }
                    let m = sc.iter().copied().fold(f32::MIN, f32::max);
                    let mut sum = 0f32;
                    for x in sc.iter_mut() {
                        *x = (*x - m).exp();
                        sum += *x;
                    }
                    for x in sc.iter_mut() {
                        *x /= sum;
                    }
                    for e in 0..d {
                        let mut a = 0f32;
                        for (jx, &j) in js.iter().enumerate() {
                            a += sc[jx] * vv[j * d + e];
                        }
                        want[i * d + e] = a;
                    }
                }
                want
            };
            let q = Tensor::from_shape(&[r, d], &qv)?.into_device()?;
            let k = Tensor::from_shape(&[d, c], &kdc)?.into_device()?;
            let v = Tensor::from_shape(&[c, d], &vv)?.into_device()?;
            let o = unsafe { DeviceTensor::uninitialized_dt(DatumType::F32, &[r, d])? };
            dispatch_metal_mfa_attention(
                stream,
                DatumType::F32,
                (1, h, r, c, d),
                scale,
                true, // causal
                None,
                get_metal_buffer(&q),
                0,
                get_metal_buffer(&k),
                0,
                get_metal_buffer(&v),
                0,
                get_metal_buffer(&o),
                0,
            )?;
            let got = o.to_host()?.into_tensor();
            let gv = unsafe { got.as_slice_unchecked::<f32>() };
            let cmp = |w: &[f32]| {
                gv.iter().zip(w.iter()).map(|(&g, &x)| (g - x).abs()).fold(0f32, f32::max)
            };
            let lower = cmp(&reference(&|i, j| j <= i));
            let upper = cmp(&reference(&|i, j| j >= i));
            let full = cmp(&reference(&|_, _| true));
            println!(
                "  causal probe: lower(j<=i)={lower:.5} upper(j>=i)={upper:.5} full={full:.5}"
            );
            // Confirms the finding: triangular-const-alone == full (unmasked) attention.
            ensure!(full < 1e-3, "expected triangular-const-alone == full attn, got full={full}");
            ensure!(lower > 0.1, "triangular const unexpectedly applied a causal mask");
            println!("FINDING confirmed: triangular const alone is a no-op (== full attention)");
            Ok(())
        })
    }

    // Additive mask path (masked=true, buffer 12). Probe with a causal mask laid out
    // [R,C]: matching `lower` => mask works + layout is [R,C] + causal-via-mask works.
    #[test]
    fn test_mfa_attention_masked() -> TractResult<()> {
        use crate::utils::get_metal_buffer;
        use tract_gpu::tensor::{DeviceTensor, IntoDevice};
        with_borrowed_metal_stream(|stream| {
            let (h, s, d) = (1usize, 64usize, 64usize);
            let (r, c) = (s, s);
            let scale = 1.0f32 / (d as f32).sqrt();
            let rng = |n: usize, seed: u64| -> Vec<f32> {
                let mut st = seed;
                (0..n)
                    .map(|_| {
                        st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        ((st >> 40) as f32 / (1u64 << 24) as f32) - 0.5
                    })
                    .collect()
            };
            let qv = rng(r * d, 1);
            let kdc = rng(d * c, 2);
            let vv = rng(c * d, 3);
            // additive causal mask, [R,C] row-major: 0 if j<=i else big-negative
            let mut mask = vec![0f32; r * c];
            for i in 0..r {
                for j in 0..c {
                    if j > i {
                        mask[i * c + j] = -1e30;
                    }
                }
            }
            let reference = |keep: &dyn Fn(usize, usize) -> bool| -> Vec<f32> {
                let mut want = vec![0f32; r * d];
                for i in 0..r {
                    let js: Vec<usize> = (0..c).filter(|&j| keep(i, j)).collect();
                    let mut sc = vec![0f32; js.len()];
                    for (jx, &j) in js.iter().enumerate() {
                        let mut a = 0f32;
                        for dd in 0..d {
                            a += qv[i * d + dd] * kdc[dd * c + j];
                        }
                        sc[jx] = a * scale;
                    }
                    let m = sc.iter().copied().fold(f32::MIN, f32::max);
                    let mut sum = 0f32;
                    for x in sc.iter_mut() {
                        *x = (*x - m).exp();
                        sum += *x;
                    }
                    for x in sc.iter_mut() {
                        *x /= sum;
                    }
                    for e in 0..d {
                        let mut a = 0f32;
                        for (jx, &j) in js.iter().enumerate() {
                            a += sc[jx] * vv[j * d + e];
                        }
                        want[i * d + e] = a;
                    }
                }
                want
            };
            let q = Tensor::from_shape(&[r, d], &qv)?.into_device()?;
            let k = Tensor::from_shape(&[d, c], &kdc)?.into_device()?;
            let v = Tensor::from_shape(&[c, d], &vv)?.into_device()?;
            let mt = Tensor::from_shape(&[r, c], &mask)?.into_device()?;
            let o = unsafe { DeviceTensor::uninitialized_dt(DatumType::F32, &[r, d])? };
            dispatch_metal_mfa_attention(
                stream,
                DatumType::F32,
                (1, h, r, c, d),
                scale,
                false,
                Some(get_metal_buffer(&mt)),
                get_metal_buffer(&q),
                0,
                get_metal_buffer(&k),
                0,
                get_metal_buffer(&v),
                0,
                get_metal_buffer(&o),
                0,
            )?;
            let got = o.to_host()?.into_tensor();
            let gv = unsafe { got.as_slice_unchecked::<f32>() };
            let cmp = |w: &[f32]| {
                gv.iter().zip(w.iter()).map(|(&g, &x)| (g - x).abs()).fold(0f32, f32::max)
            };
            let lower = cmp(&reference(&|i, j| j <= i));
            let upper = cmp(&reference(&|i, j| j >= i));
            let full = cmp(&reference(&|_, _| true));
            println!(
                "  mask([R,C] causal) probe: lower={lower:.5} upper={upper:.5} full={full:.5}"
            );
            ensure!(
                lower < 1e-2 || upper < 1e-2,
                "additive mask not applied (lower={lower} upper={upper} full={full})"
            );
            let layout = if lower < upper { "[R,C]" } else { "[C,R]" };
            println!("MFA additive mask works (causal via mask), layout {layout} ✓");
            Ok(())
        })
    }

    // PR #2 integration: tract head-major [H,S,D] layout through the on-device adapter.
    #[test]
    fn test_mfa_attention_head_major() -> TractResult<()> {
        use tract_gpu::tensor::{DeviceTensor, IntoDevice};
        with_borrowed_metal_stream(|stream| {
            let (h, r, c, d) = (2usize, 48usize, 64usize, 32usize);
            let scale = 1.0f32 / (d as f32).sqrt();
            let rng = |n: usize, seed: u64| -> Vec<f32> {
                let mut s = seed;
                (0..n)
                    .map(|_| {
                        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
                    })
                    .collect()
            };
            let qv = rng(h * r * d, 11); // [H,R,D]
            let kv = rng(h * c * d, 12); // [H,C,D]
            let vv = rng(h * c * d, 13); // [H,C,D]
            let mut want = vec![0f32; h * r * d];
            for hh in 0..h {
                for i in 0..r {
                    let mut sc = vec![0f32; c];
                    for j in 0..c {
                        let mut a = 0f32;
                        for dd in 0..d {
                            a += qv[(hh * r + i) * d + dd] * kv[(hh * c + j) * d + dd];
                        }
                        sc[j] = a * scale;
                    }
                    let m = sc.iter().copied().fold(f32::MIN, f32::max);
                    let mut sum = 0f32;
                    for x in sc.iter_mut() {
                        *x = (*x - m).exp();
                        sum += *x;
                    }
                    for x in sc.iter_mut() {
                        *x /= sum;
                    }
                    for e in 0..d {
                        let mut a = 0f32;
                        for j in 0..c {
                            a += sc[j] * vv[(hh * c + j) * d + e];
                        }
                        want[(hh * r + i) * d + e] = a;
                    }
                }
            }
            let q = Tensor::from_shape(&[h, r, d], &qv)?.into_device()?;
            let k = Tensor::from_shape(&[h, c, d], &kv)?.into_device()?;
            let v = Tensor::from_shape(&[h, c, d], &vv)?.into_device()?;
            let out = unsafe { DeviceTensor::uninitialized_dt(DatumType::F32, &[h, r, d])? };
            mfa_attention_head_major(
                stream,
                DatumType::F32,
                scale,
                false,
                None,
                &q,
                &k,
                &v,
                (h, r, c, d),
                &out,
            )?;
            let got = out.to_host()?.into_tensor();
            let gv = unsafe { got.as_slice_unchecked::<f32>() };
            let max_abs =
                gv.iter().zip(want.iter()).map(|(&g, &w)| (g - w).abs()).fold(0f32, f32::max);
            println!("  tract head-major [H,S,D] adapter: max_abs={max_abs:.6}");
            ensure!(max_abs < 1e-3, "adapter mismatch: {max_abs}");
            println!("MFA attention tract-layout adapter: matches host reference ✓");
            Ok(())
        })
    }

    // Perf signal: fused MFA attention vs the matmul lower-bound of the explode
    // path (QK^T gemm + PV gemm, softmax + S×S round-trip omitted → conservative).
    //   cargo test -p tract-metal mfa::tests::bench_mfa_attention_f32 -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_mfa_attention_f32() -> TractResult<()> {
        use crate::kernels::matmul::GemmImpl;
        use crate::utils::get_metal_buffer;
        use std::time::Instant;
        use tract_gpu::tensor::{DeviceTensor, IntoDevice};
        with_borrowed_metal_stream(|stream| {
            let z = |shape: &[usize]| -> TractResult<DeviceTensor> {
                Tensor::zero::<f32>(shape)?.into_device()
            };
            let n = 50;
            let bench = |f: &dyn Fn() -> TractResult<()>| -> TractResult<f64> {
                for _ in 0..5 {
                    f()?;
                }
                stream.wait_until_completed()?;
                let mut best = f64::MAX;
                for _ in 0..5 {
                    let t = Instant::now();
                    for _ in 0..n {
                        f()?;
                    }
                    stream.wait_until_completed()?;
                    best = best.min(t.elapsed().as_secs_f64() / n as f64);
                }
                Ok(best)
            };
            let gemm = GemmImpl::<MfaGemm>::default();
            let (h, d) = (8usize, 64usize);
            println!(
                "\n  fused MFA attention vs main explode path (QK^T+softmax+PV), f32, H={h} D={d}"
            );
            println!("  {:>6} | {:>10} | {:>12} | {:>5}", "S", "fused ms", "explode ms", "gain");
            for s_len in [128usize, 256, 512, 1024, 2048] {
                let (r, c) = (s_len, s_len);
                let scale = 1.0f32 / (d as f32).sqrt();
                let (q, k, v) = (z(&[r, h, d])?, z(&[h, d, c])?, z(&[c, h, d])?);
                let o = unsafe { DeviceTensor::uninitialized_dt(DatumType::F32, &[r, h, d])? };
                let (qg, kg, vg) = (z(&[h, r, d])?, z(&[h, d, c])?, z(&[h, c, d])?);
                let smat = unsafe { DeviceTensor::uninitialized_dt(DatumType::F32, &[h, r, c])? };
                let sm = unsafe { DeviceTensor::uninitialized_dt(DatumType::F32, &[h, r, c])? };
                let o2 = unsafe { DeviceTensor::uninitialized_dt(DatumType::F32, &[h, r, d])? };
                let fused = bench(&|| {
                    dispatch_metal_mfa_attention(
                        stream,
                        DatumType::F32,
                        (1, h, r, c, d),
                        scale,
                        false,
                        None,
                        get_metal_buffer(&q),
                        0,
                        get_metal_buffer(&k),
                        0,
                        get_metal_buffer(&v),
                        0,
                        get_metal_buffer(&o),
                        0,
                    )
                })?;
                let explode = bench(&|| {
                    gemm.dispatch_eval(stream, &qg, &kg, &smat)?;
                    crate::kernels::nn::Softmax.dispatch_eval(stream, &smat, 2, &sm)?;
                    gemm.dispatch_eval(stream, &sm, &vg, &o2)?;
                    Ok(())
                })?;
                println!(
                    "  {:>6} | {:>10.3} | {:>12.3} | {:>4.2}x",
                    s_len,
                    fused * 1e3,
                    explode * 1e3,
                    explode / fused
                );
            }
            Ok(())
        })
    }

    // Slice C: the MetalMfaSdpa op end-to-end via eval_with_session on device tensors.
    #[test]
    fn test_metal_mfa_sdpa_op() -> TractResult<()> {
        use tract_gpu::tensor::{DeviceTensorExt, IntoDevice};
        with_borrowed_metal_stream(|_stream| {
            let rng = |n: usize, seed: u64| -> Vec<f32> {
                let mut st = seed;
                (0..n)
                    .map(|_| {
                        st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        ((st >> 40) as f32 / (1u64 << 24) as f32) - 0.5
                    })
                    .collect()
            };
            let run = |is_causal: bool,
                       b: usize,
                       nh: usize,
                       sq: usize,
                       sk: usize,
                       d: usize|
             -> TractResult<f32> {
                let scale = 1.0f32 / (d as f32).sqrt();
                let qv = rng(b * nh * sq * d, 1);
                let kv = rng(b * nh * sk * d, 2);
                let vv = rng(b * nh * sk * d, 3);
                let mut want = vec![0f32; b * nh * sq * d];
                for bh in 0..b * nh {
                    for i in 0..sq {
                        let lim = if is_causal { (i + 1 + sk - sq).min(sk) } else { sk };
                        let mut sc = vec![0f32; lim];
                        for j in 0..lim {
                            let mut a = 0f32;
                            for dd in 0..d {
                                a += qv[(bh * sq + i) * d + dd] * kv[(bh * sk + j) * d + dd];
                            }
                            sc[j] = a * scale;
                        }
                        let m = sc.iter().copied().fold(f32::MIN, f32::max);
                        let mut sum = 0f32;
                        for x in sc.iter_mut() {
                            *x = (*x - m).exp();
                            sum += *x;
                        }
                        for x in sc.iter_mut() {
                            *x /= sum;
                        }
                        for e in 0..d {
                            let mut a = 0f32;
                            for j in 0..lim {
                                a += sc[j] * vv[(bh * sk + j) * d + e];
                            }
                            want[(bh * sq + i) * d + e] = a;
                        }
                    }
                }
                let qd = Tensor::from_shape(&[b, nh, sq, d], &qv)?.into_device()?;
                let kd = Tensor::from_shape(&[b, nh, sk, d], &kv)?.into_device()?;
                let vd = Tensor::from_shape(&[b, nh, sk, d], &vv)?.into_device()?;
                let op = MetalMfaSdpa { scale, is_causal };
                let out = op.eval_with_session(
                    0,
                    &TurnState::default(),
                    tvec![
                        qd.into_tensor().into_tvalue(),
                        kd.into_tensor().into_tvalue(),
                        vd.into_tensor().into_tvalue()
                    ],
                )?;
                let got = out[0].to_device_tensor()?.to_host()?.into_tensor();
                let gv = unsafe { got.as_slice_unchecked::<f32>() };
                Ok(gv.iter().zip(want.iter()).map(|(&g, &w)| (g - w).abs()).fold(0f32, f32::max))
            };
            let e1 = run(false, 1, 2, 48, 64, 32)?; // non-causal, cross (Sq!=Sk), multi-head
            println!("  op non-causal B1 H2 Sq48 Sk64 D32: max_abs={e1:.6}");
            ensure!(e1 < 1e-3, "op non-causal mismatch {e1}");
            let e2 = run(true, 1, 1, 64, 64, 64)?; // causal self-attention
            println!("  op causal B1 H1 S64 D64: max_abs={e2:.6}");
            ensure!(e2 < 1e-3, "op causal mismatch {e2}");
            println!("MetalMfaSdpa op: matches host reference (non-causal + causal) ✓");
            Ok(())
        })
    }

    // Overhead of the on-device layout adapter (4 copy_nd permutes: Q,K,V in + O out).
    #[test]
    #[ignore]
    fn bench_mfa_adapter_overhead() -> TractResult<()> {
        use std::time::Instant;
        use tract_gpu::tensor::{DeviceTensor, IntoDevice};
        with_borrowed_metal_stream(|stream| {
            let (h, r, c, d) = (8usize, 1024usize, 1024usize, 64usize);
            let z = |s: &[usize]| -> TractResult<DeviceTensor> {
                Tensor::zero::<f32>(s)?.into_device()
            };
            let (qh, kh, vh) = (z(&[h, r, d])?, z(&[h, c, d])?, z(&[h, c, d])?);
            let qn = unsafe { DeviceTensor::uninitialized_dt(DatumType::F32, &[r, h, d])? };
            let kn = unsafe { DeviceTensor::uninitialized_dt(DatumType::F32, &[h, d, c])? };
            let vn = unsafe { DeviceTensor::uninitialized_dt(DatumType::F32, &[c, h, d])? };
            let on = unsafe { DeviceTensor::uninitialized_dt(DatumType::F32, &[r, h, d])? };
            let oo = unsafe { DeviceTensor::uninitialized_dt(DatumType::F32, &[h, r, d])? };
            let cp = |inp: &DeviceTensor,
                      ins: &[usize],
                      out: &DeviceTensor,
                      outs: &[usize],
                      pick: &[usize]|
             -> TractResult<()> {
                let inn = Tensor::natural_strides(ins);
                let is: Vec<isize> = pick.iter().map(|&a| inn[a]).collect();
                let os = Tensor::natural_strides(outs);
                crate::kernels::array::metal_copy_nd_dispatch(inp, 0, &is, out, 0, outs, &os)
            };
            let n = 50;
            let bench = |f: &dyn Fn() -> TractResult<()>| -> TractResult<f64> {
                for _ in 0..5 {
                    f()?;
                }
                stream.wait_until_completed()?;
                let mut best = f64::MAX;
                for _ in 0..5 {
                    let t = Instant::now();
                    for _ in 0..n {
                        f()?;
                    }
                    stream.wait_until_completed()?;
                    best = best.min(t.elapsed().as_secs_f64() / n as f64);
                }
                Ok(best)
            };
            let permutes = bench(&|| {
                cp(&qh, &[h, r, d], &qn, &[r, h, d], &[1, 0, 2])?;
                cp(&kh, &[h, c, d], &kn, &[h, d, c], &[0, 2, 1])?;
                cp(&vh, &[h, c, d], &vn, &[c, h, d], &[1, 0, 2])?;
                cp(&on, &[r, h, d], &oo, &[h, r, d], &[1, 0, 2])?;
                Ok(())
            })?;
            println!(
                "\n  H={h} S={r} D={d}: 4 layout permutes (Q,K,V in + O out): {:.3} ms/iter",
                permutes * 1e3
            );
            println!(
                "  (fused attn ~0.92ms + permutes ~{:.2}ms = adapter; explode ~2.13ms)",
                permutes * 1e3
            );
            Ok(())
        })
    }

    // Task 2.0: dump the authoritative entry-point names from the vendored
    // libMetalFlashAttention metallib. Run:
    //   cargo test -p tract-metal mfa::tests::dump_mfa_function_names -- --ignored --nocapture
    #[test]
    #[ignore]
    fn dump_mfa_function_names() {
        use crate::kernels::{LibraryContent, LibraryName};
        let dev = metal::Device::system_default().expect("no metal device");
        let LibraryContent::Data(bytes) = LibraryName::MfaLib.content() else {
            panic!("MfaLib is not embedded data");
        };
        let lib = dev.new_library_with_data(bytes).expect("load metallib");
        let mut names = lib.function_names();
        names.sort();
        println!("\n=== MfaLib function_names ({}) ===", names.len());
        for n in &names {
            println!("  {n}");
        }
    }
}
