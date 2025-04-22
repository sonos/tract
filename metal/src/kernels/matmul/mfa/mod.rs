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
        ensure!(lhs_m1 == 1 && lhs_m2 == m, "Invalid left matmul argument [{lhs_m2}, {lhs_m1}] != [{m}, 1], strides: {:?} {:?} dims: (m: {m}, n: {n}, k: {k})",
            lhs_stride, rhs_stride);
    } else {
        // (m, k)
        ensure!(lhs_m1 == 1 && lhs_m2 == k, "Invalid left matmul argument [{lhs_m2}, {lhs_m1}] != [{k}, 1], strides: {:?} {:?} dims: (m: {m}, n: {n}, k: {k})",
            lhs_stride, rhs_stride);
    }

    if b_trans {
        // (n, k)
        ensure!(rhs_m1 == 1 && rhs_m2 == k, "Invalid right matmul argument [{rhs_m2}, {rhs_m1}] != [{k}, 1], strides: {:?} {:?} dims: (m: {m}, n: {n}, k: {k})",
            lhs_stride, rhs_stride);
    } else {
        // (k, n)
        ensure!(rhs_m1 == 1 && rhs_m2 == n, "Invalid right matmul argument [{rhs_m2}, {rhs_m1}] != [{n}, 1] {:?} {:?} dims: (m: {m}, n: {n}, k: {k})",
            lhs_stride, rhs_stride);
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
}
