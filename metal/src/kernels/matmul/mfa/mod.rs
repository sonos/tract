use crate::kernels::matmul::GemmKernel;
use crate::{ConstantValues, LibraryName, MetalContext, Value};
use anyhow::{bail, ensure, Result};
use metal::{Buffer, MTLSize, NSUInteger};
use std::ffi::c_void;
use std::fmt;
use tract_core::internal::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MfaGemmPrecision {
    Single,
    Half,
}

impl MfaGemmPrecision {
    pub fn from_dt(dt: DatumType) -> Result<MfaGemmPrecision> {
        match dt {
            DatumType::F32 => Ok(Self::Single),
            DatumType::F16 => Ok(Self::Half),
            _ => bail!("Metal GEMM only support F32 or F16 tensors"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct MfaGemm;

impl fmt::Display for MfaGemm {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "MfaGemm")
    }
}

impl GemmKernel for MfaGemm {
    fn is_supported_dt(&self, dt: DatumType) -> bool {
        matches!(dt, DatumType::F32 | DatumType::F16)
    }

    fn dispatch_eval(
        &self,
        context: &MetalContext,
        dt: DatumType,
        m: usize,
        n: usize,
        k: usize,
        a_buffer: &Buffer,
        a_offset: usize,
        transpose_a: bool,
        b_buffer: &Buffer,
        b_offset: usize,
        transpose_b: bool,
        c_buffer: &Buffer,
        c_offset: usize,
    ) -> TractResult<()> {
        let a_strides =
            if transpose_a { natural_strides(&[1, k, m]) } else { natural_strides(&[1, m, k]) };
        let b_strides =
            if transpose_b { natural_strides(&[1, n, k]) } else { natural_strides(&[1, k, n]) };

        dispatch_metal_mfa_gemm(
            context,
            MfaGemmPrecision::from_dt(dt)?,
            (1, m, n, k),
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

#[allow(clippy::too_many_arguments)]
pub fn dispatch_metal_mfa_gemm(
    context: &MetalContext,
    precision: MfaGemmPrecision,
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
) -> Result<()> {
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

    let name = match precision {
        MfaGemmPrecision::Single => "sgemm",
        MfaGemmPrecision::Half => "hgemm",
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
        MfaGemmPrecision::Single => 4,
        MfaGemmPrecision::Half => 2,
    };
    let block_bytes = block_elements * bytes;

    let command_buffer = context.command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_threadgroup_memory_length(0, block_bytes.into());
    encoder.set_buffer(0, Some(lhs_buffer), lhs_offset as NSUInteger);
    encoder.set_buffer(1, Some(rhs_buffer), rhs_offset as NSUInteger);
    encoder.set_buffer(2, Some(output), output_offset as NSUInteger);
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
    use crate::kernels::matmul::GemmImpl;
    use crate::{IntoMetal, MetalTensor};
    use derive_new::new;
    use num_traits::AsPrimitive;
    use num_traits::Float;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use tract_core::ops::einsum::BasicMatMul;

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

                let c = GemmImpl::<MfaGemm>::default().eval(context, &a, &b)?;

                let expected_c = Tensor::from_shape(
                    &[1, 2, 4],
                    &[20.0, 23.0, 26.0, 29.0, 56.0, 68.0, 80.0, 92.0],
                )?;

                let c = c.to_cpu();
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

                let c = GemmImpl::<MfaGemm>::default().eval(context, &a, &b)?;

                let expected_c = Tensor::from_shape(
                    &[2, 2, 4],
                    &[
                        20.0, 23.0, 26.0, 29.0, 56.0, 68.0, 80.0, 92.0, 344.0, 365.0, 386.0, 407.0,
                        488.0, 518.0, 548.0, 578.0,
                    ],
                )?;

                assert!(c.to_cpu().close_enough(&expected_c, Approximation::Close).is_ok());
                Ok(())
            })
        })
    }

    proptest::proptest! {
        #[test]
        fn mmm_prop_f32(pb in any::<MmmProblem<f32>>()) {
            prop_assert_eq!(pb.run().unwrap(), pb.reference().unwrap())
        }

        #[test]
        fn mmm_prop_f16(pb in any::<MmmProblem<f16>>()) {
            prop_assert_eq!(pb.run().unwrap(), pb.reference().unwrap())
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
        pub transpose_lhs: bool,
        pub rhs: Vec<F>,
        pub transpose_rhs: bool,
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
                        proptest::bool::ANY,
                        vec(rhs, rhs_len..=rhs_len),
                        proptest::bool::ANY,
                    )
                })
                .prop_map(|(b, m, k, n, lhs, transpose_lhs, rhs, transpose_rhs)| Self {
                    b,
                    m,
                    k,
                    n,
                    lhs,
                    transpose_lhs,
                    rhs,
                    transpose_rhs,
                })
                .boxed()
        }
    }

    impl<F> MmmProblem<F>
    where
        F: Datum + Float + std::ops::AddAssign,
        usize: AsPrimitive<F>,
    {
        pub fn reference(&self) -> Result<Vec<F>> {
            let matmul = BasicMatMul {
                transpose_a: self.transpose_lhs,
                transpose_b: self.transpose_rhs,
                transpose_c: false,
                quantize_output: None,
            };

            let lhs_tensor = if self.transpose_lhs {
                Tensor::from_shape(&[self.b, self.k, self.m], &self.lhs)?
            } else {
                Tensor::from_shape(&[self.b, self.m, self.k], &self.lhs)?
            };
            let rhs_tensor = if self.transpose_rhs {
                Tensor::from_shape(&[self.b, self.n, self.k], &self.rhs)?
            } else {
                Tensor::from_shape(&[self.b, self.k, self.n], &self.rhs)?
            };

            let output = matmul.eval(tvec![lhs_tensor.into_tvalue(), rhs_tensor.into_tvalue()])?;

            Ok(output[0].clone().into_tensor().as_slice::<F>()?.to_vec())
        }

        pub fn run(&self) -> Result<Vec<F>> {
            objc::rc::autoreleasepool(|| {
                crate::METAL_CONTEXT.with_borrow(|context| {
                    let lhs = if self.transpose_lhs {
                        Tensor::from_shape(&[self.b, self.k, self.m], &self.lhs)?.into_metal()?
                    } else {
                        Tensor::from_shape(&[self.b, self.m, self.k], &self.lhs)?.into_metal()?
                    };
                    let rhs = if self.transpose_rhs {
                        Tensor::from_shape(&[self.b, self.n, self.k], &self.rhs)?.into_metal()?
                    } else {
                        Tensor::from_shape(&[self.b, self.k, self.n], &self.rhs)?.into_metal()?
                    };

                    let matmul = GemmImpl::<MfaGemm>::new(self.transpose_lhs, self.transpose_rhs);

                    let c = matmul.eval(context, &lhs, &rhs)?;
                    Ok(c.to_cpu().as_slice::<F>()?.to_vec())
                })
            })
        }
    }
}
