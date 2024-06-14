use crate::MetalTensor;
use crate::{ConstantValues, LibraryName, MetalContext, Value};
use anyhow::ensure;
use anyhow::{bail, Result};
use metal::NSUInteger;
use metal::{Buffer, MTLSize};
use num_traits::Float;
use std::ffi::c_void;
use tract_core::internal::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GemmPrecision {
    Single,
    Half,
}

impl GemmPrecision {
    pub fn from_dt(dt: DatumType) -> Result<GemmPrecision> {
        match dt {
            DatumType::F32 => Ok(Self::Single),
            DatumType::F16 => Ok(Self::Half),
            _ => bail!("Metal GEMM only support F32 or F16 tensors"),
        }
    }
}

pub fn mfa_gemm_with_slice<T: Datum + Float>(
    context: &MetalContext,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs: &[T],
    lhs_strides: &[isize],
    rhs: &[T],
    rhs_strides: &[isize],
    output: &mut [T],
) -> Result<()> {
    ensure!(
        lhs_strides.len() == rhs_strides.len() && lhs_strides.len() == 3,
        "Only 3D tensors are supported in Metal GEMM"
    );

    let precision = GemmPrecision::from_dt(T::datum_type())?;

    let lhs_strides = lhs_strides.iter().map(|it| *it as usize).collect::<Vec<_>>();
    let rhs_strides = rhs_strides.iter().map(|it| *it as usize).collect::<Vec<_>>();

    let lhs_buff = context.buffer_from_slice(lhs);
    let rhs_buff = context.buffer_from_slice(rhs);
    let out_buff = context.buffer_from_slice_mut(output);
    metal_mfa_gemm(
        context,
        precision,
        (b, m, n, k),
        &lhs_strides,
        0,
        &lhs_buff,
        &rhs_strides,
        0,
        &rhs_buff,
        &out_buff,
    )?;
    context.wait_until_completed()?;
    Ok(())
}

pub fn mfa_gemm(
    context: &MetalContext,
    lhs: &MetalTensor,
    rhs: &MetalTensor,
) -> Result<MetalTensor> {
    ensure!(lhs.rank() == 3 && rhs.rank() == 3);
    ensure!(lhs.datum_type() == rhs.datum_type());

    let precision = GemmPrecision::from_dt(lhs.datum_type())?;

    let b = lhs.shape()[0];
    let m = lhs.shape()[1];
    let n = rhs.shape()[2];
    let k = lhs.shape()[2];

    let lhs_strides = lhs.strides().iter().map(|it| *it as usize).collect::<Vec<_>>();
    let rhs_strides = rhs.strides().iter().map(|it| *it as usize).collect::<Vec<_>>();

    let o_dt = lhs.datum_type();
    let o_shape = &[b, m, n];

    let output = unsafe { MetalTensor::uninitialized_dt(o_dt, o_shape)? };

    metal_mfa_gemm(
        context,
        precision,
        (b, m, n, k),
        &lhs_strides,
        0,
        lhs.metal(),
        &rhs_strides,
        0,
        rhs.metal(),
        output.metal(),
    )?;
    context.wait_until_completed()?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn metal_mfa_gemm(
    context: &MetalContext,
    precision: GemmPrecision,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs_stride: &[usize],
    lhs_offset: usize,
    lhs_buffer: &Buffer,
    rhs_stride: &[usize],
    rhs_offset: usize,
    rhs_buffer: &Buffer,
    output: &Buffer,
) -> Result<()> {
    assert!(rhs_stride.len() >= 2);
    assert!(lhs_stride.len() >= 2);
    let rhs_m1 = rhs_stride[rhs_stride.len() - 1];
    let rhs_m2 = rhs_stride[rhs_stride.len() - 2];
    let lhs_m1 = lhs_stride[lhs_stride.len() - 1];
    let lhs_m2 = lhs_stride[lhs_stride.len() - 2];
    let a_trans = if lhs_m1 == 1 && lhs_m2 == k {
        false
    } else if lhs_m1 == m && lhs_m2 == 1 {
        true
    } else {
        bail!(format!(
            "Invalid left matmul argument {:?} {:?} ({m}, {n}, {k})",
            lhs_stride, rhs_stride
        ))
    };
    let b_trans = if rhs_m1 == 1 && rhs_m2 == n {
        false
    } else if rhs_m1 == k && rhs_m2 == 1 {
        true
    } else {
        bail!(format!(
            "Invalid right matmul arguments {:?} {:?} ({m}, {n}, {k})",
            lhs_stride, rhs_stride
        ))
    };
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

    let name = match precision {
        GemmPrecision::Single => "sgemm",
        GemmPrecision::Half => "hgemm",
    };

    let pipeline = context.shared_context().load_pipeline_with_constants(
        LibraryName::MfaLib,
        name,
        constants,
    )?;
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
    let bytes = match precision {
        GemmPrecision::Single => 4,
        GemmPrecision::Half => 2,
    };
    let block_bytes = block_elements * bytes;

    let command_buffer = context.command_buffer()?;
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_threadgroup_memory_length(0, block_bytes.into());
    encoder.set_buffer(0, Some(lhs_buffer), lhs_offset as NSUInteger);
    encoder.set_buffer(1, Some(rhs_buffer), rhs_offset as NSUInteger);
    encoder.set_buffer(2, Some(output), 0);
    // TODO Tensor D

    let grid_z = b;
    if batched {
        let byte_stride_a: usize = lhs_stride[lhs_stride.len() - 3] * bytes as usize;
        let byte_stride_b: usize = rhs_stride[rhs_stride.len() - 3] * bytes as usize;
        let byte_stride_c = m * n * bytes as usize;
        // TODO byte_stride_d
        let byte_stride_d = 0;

        let buffer: Vec<u64> =
            vec![byte_stride_a as _, byte_stride_b as _, byte_stride_c as _, byte_stride_d as _];
        encoder.set_bytes(
            10,
            (buffer.len() * core::mem::size_of::<u64>()) as NSUInteger,
            buffer.as_ptr() as *const NSUInteger as *const c_void,
        );
    }

    let grid_size = MTLSize {
        width: crate::utils::div_ceil(n, n_group.into()),
        height: crate::utils::div_ceil(m, m_group.into()),
        depth: grid_z as NSUInteger,
    };
    let group_size =
        MTLSize { width: 32 * (m_splits as u64) * (n_splits as u64), height: 1, depth: 1 };
    encoder.use_resource(lhs_buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(rhs_buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(grid_size, group_size);
    encoder.end_encoding();

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IntoMetal;
    use derive_new::new;
    use num_traits::AsPrimitive;
    use proptest::collection::vec;
    use proptest::prelude::*;

    #[test]
    fn test_mfa_gemm() -> Result<()> {
        objc::rc::autoreleasepool(|| {
            crate::METAL_CONTEXT.with_borrow(|context| {
                let (b, m, n, k) = (1, 2, 4, 3);
                let a = Tensor::from_shape(
                    &[b, m, k],
                    &(0..b * m * k).map(|f| f as f32).collect::<Vec<_>>(),
                )?
                .into_metal()?;
                let b = Tensor::from_shape(
                    &[b, k, n],
                    &(0..b * n * k).map(|f| f as f32).collect::<Vec<_>>(),
                )?
                .into_metal()?;

                let c = mfa_gemm(&context, &a, &b)?;

                let expected_c = Tensor::from_shape(
                    &[1, 2, 4],
                    &[20.0, 23.0, 26.0, 29.0, 56.0, 68.0, 80.0, 92.0],
                )?;

                let c = c.into_tensor();
                assert!(c.close_enough(&expected_c, Approximation::Close).is_ok());

                let (b, m, n, k) = (2, 2, 4, 3);
                let a = MetalTensor::from_shape(
                    &[b, m, k],
                    &(0..b * m * k).map(|f| f as f32).collect::<Vec<_>>(),
                )?;
                let b = MetalTensor::from_shape(
                    &[b, k, n],
                    &(0..b * n * k).map(|f| f as f32).collect::<Vec<_>>(),
                )?;

                let c = mfa_gemm(&context, &a, &b)?;

                let expected_c = Tensor::from_shape(
                    &[2, 2, 4],
                    &[
                        20.0, 23.0, 26.0, 29.0, 56.0, 68.0, 80.0, 92.0, 344.0, 365.0, 386.0, 407.0,
                        488.0, 518.0, 548.0, 578.0,
                    ],
                )?;

                assert!(c.into_tensor().close_enough(&expected_c, Approximation::Close).is_ok());
                Ok(())
            })
        })
    }

    proptest::proptest! {
        #[test]
        fn mmm_prop_f32(pb in any::<MmmProblem<f32>>()) {
            prop_assert_eq!(pb.run().unwrap(), pb.reference())
        }

        #[test]
        fn mmm_prop_f16(pb in any::<MmmProblem<f16>>()) {
            prop_assert_eq!(pb.run().unwrap(), pb.reference())
        }
    }

    #[derive(Debug, new)]
    pub struct MmmProblem<F: Datum + Float>
    where
        F: Datum + Float,
        usize: AsPrimitive<F>,
    {
        pub b: usize,
        pub m: usize,
        pub k: usize,
        pub n: usize,
        pub lhs: Vec<F>,
        pub rhs: Vec<F>,
    }

    impl<F> Arbitrary for MmmProblem<F>
    where
        F: Datum + Float,
        usize: AsPrimitive<F>,
    {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(_: ()) -> Self::Strategy {
            (1usize..2, 1usize..20, 1usize..20, 1usize..20)
                .prop_flat_map(|(b, m, k, n)| {
                    let lhs_len = b * m * k;
                    let rhs_len = b * k * n;
                    let lhs = (0usize..10).prop_map(|x| x.as_());
                    let rhs = (0usize..10).prop_map(|x| x.as_());
                    (
                        Just(b),
                        Just(m),
                        Just(k),
                        Just(n),
                        vec(lhs, lhs_len..=lhs_len),
                        vec(rhs, rhs_len..=rhs_len),
                    )
                })
                .prop_map(|(b, m, k, n, lhs, rhs)| Self { b, m, k, n, lhs, rhs })
                .boxed()
        }
    }

    impl<F> MmmProblem<F>
    where
        F: Datum + Float + std::ops::AddAssign,
        usize: AsPrimitive<F>,
    {
        pub fn reference(&self) -> Vec<F> {
            let mut vi = vec![F::zero(); self.b * self.m * self.n];
            for m in 0..self.m {
                for n in 0..self.n {
                    for k in 0..self.k {
                        // m, k * k, n
                        let lhs: F = self.lhs[k + self.k * m];
                        let rhs: F = self.rhs[n + self.n * k];
                        let offset = n + m * self.n;
                        vi[offset] += lhs * rhs;
                    }
                }
            }
            vi
        }

        pub fn run(&self) -> Result<Vec<F>> {
            objc::rc::autoreleasepool(|| {
                crate::METAL_CONTEXT.with_borrow(|context| {
                    let (b, m, n, k) = dbg!((self.b, self.m, self.n, self.k));
                    let lhs = Tensor::from_shape(&[b, m, k], &self.lhs)?.into_metal()?;
                    let rhs = Tensor::from_shape(&[b, k, n], &self.rhs)?.into_metal()?;
                    let c = mfa_gemm(context, &lhs, &rhs)?;
                    Ok(c.into_tensor().as_slice::<F>()?.to_vec())
                })
            })
        }
    }
}
