use crate::kernels::GemmPrecision;
use crate::{MetalContext, MetalTensor};
use anyhow::{bail, ensure, Result};
use num_traits::Float;
use tract_core::internal::{Datum, DatumType};

pub fn gemm_precision_from_dt(dt: DatumType) -> Result<GemmPrecision> {
    match dt {
        DatumType::F32 => Ok(GemmPrecision::Single),
        DatumType::F16 => Ok(GemmPrecision::Half),
        _ => bail!("Metal GEMM only support F32 or F16 tensors"),
    }
}

pub fn gemm_with_slice<T: Datum + Float>(
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

    let precision = gemm_precision_from_dt(T::datum_type())?;

    let lhs_strides = lhs_strides.iter().map(|it| *it as usize).collect::<Vec<_>>();
    let rhs_strides = rhs_strides.iter().map(|it| *it as usize).collect::<Vec<_>>();

    let lhs_buff = context.buffer_from_slice_with_copy(lhs);
    let rhs_buff = context.buffer_from_slice_with_copy(rhs);
    let out_buff = context.buffer_from_slice_with_copy_mut(output);
    crate::kernels::metal_gemm(
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

pub fn gemm(context: &MetalContext, lhs: &MetalTensor, rhs: &MetalTensor) -> Result<MetalTensor> {
    ensure!(lhs.rank() == 3 && rhs.rank() == 3);
    ensure!(lhs.datum_type() == rhs.datum_type());

    let precision = gemm_precision_from_dt(lhs.datum_type())?;

    let b = lhs.shape()[0];
    let m = lhs.shape()[1];
    let n = rhs.shape()[2];
    let k = lhs.shape()[2];

    let lhs_strides = lhs.strides().iter().map(|it| *it as usize).collect::<Vec<_>>();
    let rhs_strides = rhs.strides().iter().map(|it| *it as usize).collect::<Vec<_>>();

    let o_dt = lhs.datum_type();
    let o_shape = &[b, m, n];

    let output = unsafe { MetalTensor::uninitialized_dt(o_dt, o_shape)? };

    crate::kernels::metal_gemm(
        context,
        precision,
        (b, m, n, k),
        &lhs_strides,
        0,
        &lhs.metal(),
        &rhs_strides,
        0,
        &rhs.metal(),
        output.metal(),
    )?;
    context.wait_until_completed()?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IntoMetal;
    use derive_new::new;
    use num_traits::AsPrimitive;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use tract_core::internal::*;

    #[test]
    fn test_gemm() -> Result<()> {
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

                let c = gemm(&context, &a, &b)?;

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

                let c = gemm(&context, &a, &b)?;

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
                    let c = gemm(context, &lhs, &rhs)?;
                    Ok(c.into_tensor().as_slice::<F>()?.to_vec())
                })
            })
        }
    }
}
