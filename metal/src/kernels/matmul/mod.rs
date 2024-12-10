mod basic;
mod mfa;
mod mlx_gemm;
mod mmm_tile_8x8;
pub mod mps;

pub use basic::BasicMatMul;
pub use mfa::MfaGemm;
pub use mlx_gemm::MlxGemm;
pub use mmm_tile_8x8::{metal_mmm_tile_8x8, mmm_tile_8x8};
pub use mps::MpsMatMul;

use crate::{MetalContext, MetalTensor};
use metal::Buffer;
use num_traits::One;
use std::fmt;
use tract_core::internal::*;

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

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub struct GemmDispatchParams {
    pub dt: DatumType,
    pub batch: usize,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub transpose_a: bool,
    pub a_offset: usize,
    pub transpose_b: bool,
    pub b_offset: usize,
    pub c_offset: usize,
}

impl GemmDispatchParams {
    #[allow(clippy::too_many_arguments)]
    pub fn compute_dispatches_params(
        dt: DatumType,
        a_offset: usize,
        a_shape: &[usize],
        transpose_a: bool,
        b_offset: usize,
        b_shape: &[usize],
        transpose_b: bool,
        c_offset: usize,
        c_shape: &[usize],
    ) -> TractResult<Vec<GemmDispatchParams>> {
        let rank = c_shape.len();
        let squeezed_a_shape = squeeze_batch_axes(a_shape)?;
        let squeezed_b_shape = squeeze_batch_axes(b_shape)?;
        let squeezed_c_shape = squeeze_batch_axes(c_shape)?;

        let a_batch = squeezed_a_shape[0];
        let b_batch = squeezed_b_shape[0];

        ensure!(squeezed_c_shape[0] == a_batch || squeezed_c_shape[0] == b_batch);

        let m = c_shape[rank - 2];
        let n = c_shape[rank - 1];
        let k = a_shape[a_shape.len() - 2 + !transpose_a as usize];

        match (a_batch, b_batch) {
            // bmk, 1kn -> bmn
            // bmk, 1nk -> bmn
            (a_batch, 1) if a_batch != 1 && !transpose_a => Ok(vec![GemmDispatchParams {
                dt,
                batch: 1,
                m: m * a_batch,
                n,
                k,
                transpose_a,
                a_offset,
                transpose_b,
                b_offset,
                c_offset,
            }]),
            // bkm, 1kn -> bmn
            // bkm, 1nk -> bmn
            // As many dispatches as batch dimension.
            (a_batch, 1) if a_batch != 1 => Ok((0..a_batch)
                .map(|a_batch_idx| GemmDispatchParams {
                    dt,
                    batch: 1,
                    m,
                    n,
                    k,
                    transpose_a,
                    a_offset: a_offset + a_batch_idx * m * k * dt.size_of(),
                    transpose_b,
                    b_offset,
                    c_offset: c_offset + a_batch_idx * m * n * dt.size_of(),
                })
                .collect()),
            // 1mk, bkn -> bmn
            // 1km, bkn -> bmn
            // 1mk, bnk -> bmn
            // 1km, bnk -> bmn
            // As many dispatch as batch dimension.
            (1, b_batch) if b_batch != 1 => Ok((0..b_batch)
                .map(|b_batch_idx| GemmDispatchParams {
                    dt,
                    batch: 1,
                    m,
                    n,
                    k,
                    transpose_a,
                    a_offset,
                    transpose_b,
                    b_offset: b_offset + b_batch_idx * n * k * dt.size_of(),
                    c_offset: c_offset + b_batch_idx * m * n * dt.size_of(),
                })
                .collect()),
            // bmk, bkn -> bmn
            // bkm, bkn -> bmn
            // bmk, bnk -> bmn
            // bkm, bnk -> bmn
            (a_batch, b_batch) => {
                ensure!(a_batch == b_batch);

                Ok(vec![GemmDispatchParams {
                    dt,
                    batch: a_batch,
                    m,
                    n,
                    k,
                    transpose_a,
                    a_offset,
                    transpose_b,
                    b_offset,
                    c_offset,
                }])
            }
        }
    }
}

pub trait GemmKernel: fmt::Display + fmt::Debug + Clone + Default + Send + Sync {
    fn name() -> &'static str;

    fn is_supported_dt(&self, dt: DatumType) -> bool;

    fn dispatch_eval(
        &self,
        context: &MetalContext,
        params: GemmDispatchParams,
        a_buffer: &Buffer,
        b_buffer: &Buffer,
        c_buffer: &Buffer,
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
        let c_dt = a.datum_type();
        let c_shape = self.output_shape(a.shape(), b.shape());
        let c = unsafe { MetalTensor::uninitialized_dt(c_dt, &c_shape)? };

        self.dispatch_eval(context, a, b, &c)?;
        context.wait_until_completed()?;
        Ok(c)
    }

    pub fn dispatch_eval(
        &self,
        context: &MetalContext,
        a: &MetalTensor,
        b: &MetalTensor,
        c: &MetalTensor,
    ) -> TractResult<()> {
        a.retain_until_completion();
        b.retain_until_completion();
        c.retain_until_completion();

        ensure!(c.datum_type() == a.datum_type());
        ensure!(c.shape() == self.output_shape(a.shape(), b.shape()).as_slice());

        if c.shape().iter().product::<usize>() == 0 {
            return Ok(());
        }

        let dispatches = GemmDispatchParams::compute_dispatches_params(
            c.datum_type(),
            a.metal_offset(),
            a.shape(),
            self.transpose_a,
            b.metal_offset(),
            b.shape(),
            self.transpose_b,
            c.metal_offset(),
            c.shape(),
        )?;

        for d in dispatches {
            self.matmul
                .dispatch_eval(
                    context,
                    d,
                    a.metal(),
                    b.metal(),
                    c.metal(),
                )
                .with_context(|| {
                    anyhow!(
                    "Error while performing MatMul with {:?} (a: {:?}), (b: {:?}) = (c: {:?}) for dispatch: {:?}",
                    self.matmul,
                    a.shape(),
                    b.shape(),
                    c.shape(),
                    d,
                )
            })?;
        }

        Ok(())
    }
}

// Squeeze batch axes and return a shape with a rank of 3.
fn squeeze_batch_axes(s: &[usize]) -> TractResult<TVec<usize>> {
    ensure!(s.len() >= 2);
    let rank = s.len();
    if s.len() == 2 {
        return Ok(tvec![1, s[rank - 2], s[rank - 1]]);
    }
    let rank = s.len();
    Ok(tvec![s[..rank - 2].iter().product(), s[rank - 2], s[rank - 1],])
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

    pub(crate) fn run_mmm_test_case<K: GemmKernel>(
        (b, m, k, n): (usize, usize, usize, usize),
        transpose_a: bool,
        transpose_b: bool,
    ) -> TractResult<()> {
        objc::rc::autoreleasepool(|| {
            crate::METAL_CONTEXT.with_borrow(|context| {
                let a_shape = if !transpose_a { [b, m, k] } else { [b, k, m] };
                let b_shape = if !transpose_b { [b, k, n] } else { [b, n, k] };
                let a = Tensor::from_shape(
                    &a_shape,
                    &(0..b * m * k).map(|f| f as f32 / 100.0).collect::<Vec<_>>(),
                )?
                .into_metal()?;

                let b = Tensor::from_shape(
                    &b_shape,
                    &(0..b * k * n).map(|f| f as f32 / 100.0).collect::<Vec<_>>(),
                )?
                .into_metal()?;

                let metal_output =
                    GemmImpl::<K>::new(transpose_a, transpose_b).eval(context, &a, &b)?;
                let matmul = BasicMatMul {
                    transpose_a,
                    transpose_b,
                    transpose_c: false,
                    quantize_output: None,
                };
                let output = args_1!(
                    matmul.eval(tvec![a.to_cpu()?.into_tvalue(), b.to_cpu()?.into_tvalue()])?
                );
                output.close_enough(&metal_output.to_cpu()?, Approximation::Approximate)?;
                Ok(())
            })
        })
    }

    #[test]
    fn test_gemm_dispatches_params() -> Result<()> {
        let dt = DatumType::F32;
        let (m, k, n) = (2, 3, 4);
        assert_eq!(
            GemmDispatchParams::compute_dispatches_params(
                dt,
                0,
                &[1, m, k],
                false,
                0,
                &[1, k, n],
                false,
                0,
                &[1, m, n],
            )?,
            vec![GemmDispatchParams {
                dt,
                batch: 1,
                m,
                n,
                k,
                transpose_a: false,
                a_offset: 0,
                transpose_b: false,
                b_offset: 0,
                c_offset: 0,
            }]
        );

        assert_eq!(
            GemmDispatchParams::compute_dispatches_params(
                dt,
                0,
                &[10, m, k],
                false,
                0,
                &[10, k, n],
                false,
                0,
                &[10, m, n],
            )?,
            vec![GemmDispatchParams {
                dt,
                batch: 10,
                m,
                n,
                k,
                transpose_a: false,
                a_offset: 0,
                transpose_b: false,
                b_offset: 0,
                c_offset: 0,
            }]
        );

        assert_eq!(
            GemmDispatchParams::compute_dispatches_params(
                dt,
                0,
                &[1, m, k],
                false,
                0,
                &[2, k, n],
                false,
                10,
                &[2, m, n],
            )?,
            vec![
                GemmDispatchParams {
                    dt,
                    batch: 1,
                    m,
                    n,
                    k,
                    transpose_a: false,
                    a_offset: 0,
                    transpose_b: false,
                    b_offset: 0,
                    c_offset: 10,
                },
                GemmDispatchParams {
                    dt,
                    batch: 1,
                    m,
                    n,
                    k,
                    transpose_a: false,
                    a_offset: 0,
                    transpose_b: false,
                    b_offset: 1 * n * k * dt.size_of(),
                    c_offset: 10 + m * n * dt.size_of(),
                }
            ]
        );

        assert_eq!(
            GemmDispatchParams::compute_dispatches_params(
                dt,
                0,
                &[2, k, m],
                true,
                0,
                &[2, k, n],
                false,
                100,
                &[2, m, n],
            )?,
            vec![GemmDispatchParams {
                dt,
                batch: 2,
                m,
                n,
                k,
                transpose_a: true,
                a_offset: 0,
                transpose_b: false,
                b_offset: 0,
                c_offset: 100,
            }]
        );

        assert_eq!(
            GemmDispatchParams::compute_dispatches_params(
                dt,
                0,
                &[2, k, m],
                true,
                0,
                &[1, k, n],
                false,
                100,
                &[2, m, n],
            )?,
            vec![
                GemmDispatchParams {
                    dt,
                    batch: 1,
                    m,
                    n,
                    k,
                    transpose_a: true,
                    a_offset: 0,
                    transpose_b: false,
                    b_offset: 0,
                    c_offset: 100,
                },
                GemmDispatchParams {
                    dt,
                    batch: 1,
                    m,
                    n,
                    k,
                    transpose_a: true,
                    a_offset: 1 * m * k * dt.size_of(),
                    transpose_b: false,
                    b_offset: 0,
                    c_offset: 100 + 1 * m * n * dt.size_of(),
                }
            ]
        );

        assert_eq!(
            GemmDispatchParams::compute_dispatches_params(
                dt,
                0,
                &[10, m, k],
                false,
                10,
                &[1, k, n],
                false,
                0,
                &[10, m, n],
            )?,
            vec![GemmDispatchParams {
                dt,
                batch: 1,
                m: 10 * m,
                n,
                k,
                transpose_a: false,
                a_offset: 0,
                transpose_b: false,
                b_offset: 10,
                c_offset: 0,
            }]
        );

        Ok(())
    }

    #[test]
    fn test_squeeze_batch_axes() -> Result<()> {
        assert_eq!(squeeze_batch_axes(&[1, 2, 3, 4])?, tvec![2, 3, 4]);
        assert_eq!(squeeze_batch_axes(&[3, 2, 3, 4])?, tvec![6, 3, 4]);
        assert_eq!(squeeze_batch_axes(&[3, 1, 2, 3, 4])?, tvec![6, 3, 4]);
        assert!(squeeze_batch_axes(&[1]).is_err());
        assert_eq!(squeeze_batch_axes(&[1, 1, 3, 4])?, tvec![1, 3, 4]);
        Ok(())
    }

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
            (1usize..10, 1usize..20, 1usize..20, 1usize..20)
                .prop_flat_map(|(b, m, k, n)| {
                    let lhs_len = b * m * k;
                    let rhs_len = b * k * n;
                    let datum = (0usize..10).prop_map(|x| x.as_());
                    (
                        Just(b),
                        Just(m),
                        Just(k),
                        Just(n),
                        vec(datum.clone(), lhs_len..=lhs_len),
                        proptest::bool::ANY,
                        vec(datum, rhs_len..=rhs_len),
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
                    Ok(c.to_cpu()?.as_slice::<F>()?.to_vec())
                })
            })
        }
    }
}
