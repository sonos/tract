mod basic_mat_mul;
mod mfa;
mod mlx_gemm;
mod mmm_tile_8x8;
pub mod mps;

pub use basic_mat_mul::BasicMatMul;
pub use mfa::MfaGemm;
pub use mlx_gemm::MlxGemm;
pub use mmm_tile_8x8::{metal_mmm_tile_8x8, mmm_tile_8x8};
pub use mps::MpsMatMul;

use crate::{MetalContext, MetalTensor};
use metal::Buffer;
use num_traits::One;
use std::fmt;
use tract_core::{internal::*, ndarray, ndarray::Dimension};

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub enum MetalGemmImplKind {
    Mlx,
    Mps,
    Mfa,
}

impl Default for MetalGemmImplKind {
    fn default() -> Self {
        Self::Mlx
    }
}

pub trait GemmKernel: fmt::Display + fmt::Debug + Clone + Default + Send + Sync {
    fn is_supported_dt(&self, dt: DatumType) -> bool;

    #[allow(clippy::too_many_arguments)]
    fn dispatch_eval(
        &self,
        context: &MetalContext,
        dt: DatumType,
        m: usize,
        k: usize,
        n: usize,
        a_buffer: &Buffer,
        a_offset: usize,
        a_transpose: bool,
        b_buffer: &Buffer,
        b_offset: usize,
        b_transpose: bool,
        c_buffer: &Buffer,
        c_offset: usize,
    ) -> TractResult<()>;
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct GemmImpl<M: GemmKernel> {
    pub transpose_a: bool,
    pub transpose_b: bool,
    pub matmul: M,
}

impl<M: GemmKernel> fmt::Display for GemmImpl<M> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.matmul)
    }
}

impl<M: GemmKernel> GemmImpl<M> {
    pub fn new(transpose_a: bool, transpose_b: bool) -> Self {
        Self { transpose_a, transpose_b, matmul: M::default() }
    }

    pub fn is_supported_dt(&self, dt: DatumType) -> bool {
        self.matmul.is_supported_dt(dt)
    }

    pub fn output_shape<D: DimLike + One>(&self, a: &[D], b: &[D]) -> TVec<D> {
        let rank = a.len();
        let mut output: TVec<D> = (0..rank - 2)
            .map(|ix| if a[ix].is_one() { b[ix].clone() } else { a[ix].clone() })
            .collect();
        output.push(a[rank - 2 + self.transpose_a as usize].clone());
        output.push(b[rank - 2 + !self.transpose_b as usize].clone());
        output
    }

    pub fn eval(
        &self,
        context: &MetalContext,
        a: &MetalTensor,
        b: &MetalTensor,
    ) -> TractResult<MetalTensor> {
        let output = self.dispatch_eval(context, a, b)?;
        context.wait_until_completed()?;
        Ok(output)
    }

    pub fn dispatch_eval(
        &self,
        context: &MetalContext,
        a: &MetalTensor,
        b: &MetalTensor,
    ) -> TractResult<MetalTensor> {
        a.retain_until_completion();
        b.retain_until_completion();

        let c_dt = a.datum_type();
        let c_shape = self.output_shape(a.shape(), b.shape());

        let rank = c_shape.len();
        let m = c_shape[rank - 2];
        let n = c_shape[rank - 1];
        let k = a.shape()[a.rank() - 2 + !self.transpose_a as usize];

        unsafe {
            let c = MetalTensor::zero_dt(c_dt, &c_shape)?;
            c.retain_until_completion();

            if k == 0 {
                return Ok(c);
            }

            let silent_a_axis = c.rank() - a.rank();
            let silent_b_axis = c.rank() - b.rank();
            for prefix in ndarray::indices(&c_shape[0..rank - 2]) {
                let mut a_offset = 0;
                let mut b_offset = 0;
                let mut c_offset = 0;
                for (axis, x) in prefix.as_array_view().iter().enumerate() {
                    if axis >= silent_a_axis && a.shape()[axis - silent_a_axis] != 1 {
                        a_offset += *x as isize * a.strides()[axis - silent_a_axis];
                    }
                    if axis >= silent_b_axis && b.shape()[axis - silent_b_axis] != 1 {
                        b_offset += *x as isize * b.strides()[axis - silent_b_axis];
                    }
                    c_offset += *x as isize * c.strides()[axis];
                }

                self.matmul
                    .dispatch_eval(
                        context,
                        c_dt,
                        m,
                        n,
                        k,
                        a.metal(),
                        a_offset as usize * c_dt.size_of(),
                        self.transpose_a,
                        b.metal(),
                        b_offset as usize * c_dt.size_of(),
                        self.transpose_b,
                        c.metal(),
                        c_offset as usize * c_dt.size_of(),
                    )
                    .with_context(|| {
                        anyhow!(
                        "Error while performing MatMul with {:?} (a: {:?}), (b: {:?}) = (c: {:?})",
                        self.matmul,
                        a.shape(),
                        b.shape(),
                        c_shape
                    )
                    })?;
            }

            Ok(c)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::matmul::GemmImpl;
    use crate::IntoMetal;
    use anyhow::Result;
    use derive_new::new;
    use num_traits::AsPrimitive;
    use num_traits::Float;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use tract_core::ops::einsum::BasicMatMul;

    proptest::proptest! {
        #[test]
        fn mmm_mfa_prop_f32(pb in any::<MmmProblem<MfaGemm, f32>>()) {
            prop_assert_eq!(pb.run().unwrap(), pb.reference().unwrap())
        }

        #[test]
        fn mmm_mfa_prop_f16(pb in any::<MmmProblem<MfaGemm, f16>>()) {
            prop_assert_eq!(pb.run().unwrap(), pb.reference().unwrap())
        }

        #[test]
        fn mmm_mps_prop_f32(pb in any::<MmmProblem<MpsMatMul, f32>>()) {
            prop_assert_eq!(pb.run().unwrap(), pb.reference().unwrap())
        }

        #[test]
        fn mmm_mps_prop_f16(pb in any::<MmmProblem<MpsMatMul, f16>>()) {
            prop_assert_eq!(pb.run().unwrap(), pb.reference().unwrap())
        }

        #[test]
        fn mmm_mlx_prop_f32(pb in any::<MmmProblem<MlxGemm, f32>>()) {
            prop_assert_eq!(pb.run().unwrap(), pb.reference().unwrap())
        }

        #[test]
        fn mmm_mlx_prop_f16(pb in any::<MmmProblem<MlxGemm, f16>>()) {
            prop_assert_eq!(pb.run().unwrap(), pb.reference().unwrap())
        }
    }

    #[derive(Debug, new)]
    pub struct MmmProblem<K: GemmKernel, F: Datum + Float>
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
        pub _phantom: std::marker::PhantomData<K>,
    }

    impl<K, F> Arbitrary for MmmProblem<K, F>
    where
        K: GemmKernel,
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
                    _phantom: std::marker::PhantomData,
                })
                .boxed()
        }
    }

    impl<K, F> MmmProblem<K, F>
    where
        K: GemmKernel,
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

                    let matmul = GemmImpl::<K>::new(self.transpose_lhs, self.transpose_rhs);

                    let c = matmul.eval(context, &lhs, &rhs)?;
                    Ok(c.to_cpu().as_slice::<F>()?.to_vec())
                })
            })
        }
    }
}
