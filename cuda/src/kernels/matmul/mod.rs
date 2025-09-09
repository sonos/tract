mod ggml;

use cudarc::driver::{CudaView, CudaViewMut};
pub use ggml::GgmlGemm;
use tract_core::tract_linalg::block_quant::{BlockQuant, Q4_0};

use num_traits::One;
use std::fmt;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;
use tract_gpu::utils::as_q40_fact;

use crate::context::TractCudaStream;
use crate::kernels::{
    get_cuda_view, get_cuda_view_mut, get_sliced_cuda_view, get_sliced_cuda_view_mut,
};
use crate::utils::get_q40_fact;

#[derive(Debug, Default, PartialEq, Eq, Hash, Clone, Copy)]
pub enum CudaGemmImplKind {
    #[default]
    Ggml,
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub struct GemmDispatchParams {
    pub dts: [DatumType; 3],
    pub a_batch: usize,
    pub b_batch: usize,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub transpose_a: bool,
    pub a_offset: usize,
    pub transpose_b: bool,
    pub b_offset: usize,
    pub q40_b: bool,
    pub c_offset: usize,
    pub a_strides: TVec<isize>,
    pub b_strides: TVec<isize>,
    pub c_strides: TVec<isize>,
}

impl GemmDispatchParams {
    #[allow(clippy::too_many_arguments)]
    pub fn compute_dispatches_params<M: GemmKernel>(
        dts: [DatumType; 3],
        a_shape: &[usize],
        transpose_a: bool,
        b_shape: &[usize],
        transpose_b: bool,
        q40_b: bool,
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

        ensure!((a_batch % b_batch == 0) || (a_batch == 1));
        let a_strides = if transpose_a {
            natural_strides(&[a_batch, k, m])
        } else {
            natural_strides(&[a_batch, m, k])
        };

        let b_strides = if transpose_b {
            natural_strides(&[b_batch, n, k])
        } else {
            natural_strides(&[b_batch, k, n])
        };

        let b_batch_stride = if !q40_b {
            n * k * dts[1].size_of()
        } else {
            ensure!(k % Q4_0.block_len() == 0);
            n * (k / Q4_0.block_len()) * Q4_0.block_bytes()
        };

        let c_strides = natural_strides(&[a_batch.max(b_batch), m, n]);
        match (a_batch, b_batch) {
            // bmk, 1kn -> bmn
            // bmk, 1nk -> bmn
            (a_batch, 1) if a_batch != 1 && !transpose_a => Ok(vec![GemmDispatchParams {
                dts,
                a_batch: 1,
                b_batch: 1,
                m: m * a_batch,
                n,
                k,
                transpose_a,
                a_offset: 0,
                transpose_b,
                b_offset: 0,
                q40_b,
                c_offset: 0,
                a_strides,
                b_strides,
                c_strides,
            }]),
            // bkm, 1kn -> bmn
            // bkm, 1nk -> bmn
            // As many dispatches as batch dimension.
            (a_batch, 1) if a_batch != 1 => Ok((0..a_batch)
                .map(|a_batch_idx| GemmDispatchParams {
                    dts,
                    a_batch: 1,
                    b_batch: 1,
                    m,
                    n,
                    k,
                    transpose_a,
                    a_offset: a_batch_idx * m * k * dts[0].size_of(),
                    transpose_b,
                    b_offset: 0,
                    q40_b,
                    c_offset: a_batch_idx * m * n * dts[2].size_of(),
                    a_strides: a_strides.clone(),
                    b_strides: b_strides.clone(),
                    c_strides: c_strides.clone(),
                })
                .collect()),
            // 1mk, bkn -> bmn
            // 1km, bkn -> bmn
            // 1mk, bnk -> bmn
            // 1km, bnk -> bmn
            // As many dispatch as batch dimension.
            (1, b_batch) if b_batch != 1 => Ok((0..b_batch)
                .map(|b_batch_idx| GemmDispatchParams {
                    dts,
                    a_batch: 1,
                    b_batch: 1,
                    m,
                    n,
                    k,
                    transpose_a,
                    a_offset: 0,
                    transpose_b,
                    b_offset: b_batch_idx * b_batch_stride,
                    q40_b,
                    c_offset: b_batch_idx * m * n * dts[2].size_of(),
                    a_strides: a_strides.clone(),
                    b_strides: b_strides.clone(),
                    c_strides: c_strides.clone(),
                })
                .collect()),
            (a_batch, b_batch) => {
                if M::supports_broadcast(a_batch, b_batch, m, k, n, q40_b) || (a_batch == b_batch) {
                    Ok(vec![GemmDispatchParams {
                        dts,
                        a_batch,
                        b_batch,
                        m,
                        n,
                        k,
                        transpose_a,
                        a_offset: 0,
                        transpose_b,
                        b_offset: 0,
                        q40_b,
                        c_offset: 0,
                        a_strides,
                        b_strides,
                        c_strides,
                    }])
                } else {
                    bail!("a_batch != b_batch and backend does not support broadcast");
                }
            }
        }
    }
}

pub trait GemmKernel: fmt::Display + fmt::Debug + Clone + Default + Send + Sync {
    fn name() -> &'static str;

    fn supports_broadcast(
        _a_batch: usize,
        _b_batch: usize,
        _m: usize,
        _k: usize,
        _n: usize,
        is_q40: bool,
    ) -> bool {
        false
    }

    fn is_supported_dts(&self, facts: &[TypedFact]) -> bool {
        assert!(facts.len() == 2, "Expected 2 inputs for matmul");
        matches!(facts[0].datum_type, DatumType::F32 | DatumType::F16)
            && facts[0].datum_type == facts[1].datum_type
    }

    fn output_dt(&self, a_dt: DatumType, b_dt: DatumType) -> TractResult<DatumType> {
        ensure!([DatumType::F16, DatumType::F32].contains(&a_dt));
        ensure!(a_dt == b_dt);
        Ok(a_dt)
    }

    fn dispatch_eval(
        &self,
        stream: &TractCudaStream,
        params: GemmDispatchParams,
        a_buffer: &CudaView<'_, u8>,
        b_buffer: &CudaView<'_, u8>,
        c_buffer: &mut CudaViewMut<'_, u8>,
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

    pub fn output_shape<D: DimLike + One>(&self, a: &[D], b: &[D]) -> TVec<D> {
        let rank = a.len();
        let mut output: TVec<D> = (0..rank - 2)
            .map(|ix| if a[ix].is_one() { b[ix].clone() } else { a[ix].clone() })
            .collect();
        output.push(a[rank - 2 + self.transpose_a as usize].clone());
        output.push(b[rank - 2 + !self.transpose_b as usize].clone());
        output
    }

    pub fn output_facts(
        &self,
        shape: &[TDim],
        a_dt: DatumType,
        b_dt: DatumType,
    ) -> TractResult<TVec<TypedFact>> {
        let out_dt = self.matmul.output_dt(a_dt, b_dt)?;
        ensure!([DatumType::F16, DatumType::F32].contains(&out_dt));
        Ok(tvec!(out_dt.fact(shape)))
    }

    pub fn eval(
        &self,
        stream: &TractCudaStream,
        a: &DeviceTensor,
        b: &DeviceTensor,
    ) -> TractResult<DeviceTensor> {
        let q40_b = get_q40_fact(b);
        let b_shape = q40_b
            .clone()
            .map(|bqf| b.shape().iter().cloned().chain(bqf.shape().iter().copied()).collect())
            .unwrap_or(b.shape().to_vec());

        let c_dt = self.matmul.output_dt(a.datum_type(), b.datum_type())?;
        let c_shape = self.output_shape(a.shape(), &b_shape);
        let c = unsafe { DeviceTensor::uninitialized_dt(c_dt, &c_shape)? };

        self.dispatch_eval(stream, a, b, &c)?;
        stream.synchronize()?;
        Ok(c)
    }

    pub fn dispatch_eval(
        &self,
        stream: &TractCudaStream,
        a: &DeviceTensor,
        b: &DeviceTensor,
        c: &DeviceTensor,
    ) -> TractResult<()> {
        let q40_b = get_q40_fact(b);
        let b_shape = q40_b
            .clone()
            .map(|bqf| b.shape().iter().cloned().chain(bqf.shape().iter().copied()).collect())
            .unwrap_or(b.shape().to_vec());

        ensure!(c.shape() == self.output_shape(a.shape(), &b_shape).as_slice());

        if c.shape().iter().product::<usize>() == 0 {
            return Ok(());
        }

        let dispatches = GemmDispatchParams::compute_dispatches_params::<M>(
            [a.datum_type(), b.datum_type(), c.datum_type()],
            a.shape(),
            self.transpose_a,
            &b_shape,
            self.transpose_b,
            q40_b.is_some(),
            c.shape(),
        )?;

        for d in dispatches {
            let a_view = get_sliced_cuda_view(
                a,
                d.a_offset,
                d.a_strides[0] as usize * d.a_batch * d.dts[0].size_of(),
            )?;
            let b_len = if d.q40_b {
                d.b_strides[0] as usize * d.b_batch / Q4_0.block_len() * Q4_0.block_bytes()
            } else {
                d.b_strides[0] as usize * d.b_batch * d.dts[1].size_of()
            };
            let b_view = get_sliced_cuda_view(b, d.b_offset, b_len)?;
            let mut c_view = get_sliced_cuda_view_mut(
                c,
                d.c_offset,
                d.c_strides[0] as usize * d.a_batch.max(d.b_batch) * d.dts[2].size_of(),
            )?;
            self.matmul
                .dispatch_eval(
                    stream,
                    d.clone(),
                    &a_view,
                    &b_view,
                    &mut c_view,
                )
                .with_context(|| {
                    format!(
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
    use crate::context::CUDA_STREAM;
    use crate::kernels::matmul::GemmImpl;
    use crate::utils::pad_q40;
    use num_traits::AsPrimitive;
    use num_traits::Float;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use tract_core::ops::einsum::prefix_matmul::PrefixMatMul;
    use tract_core::tract_data::itertools::Itertools;
    use tract_core::tract_linalg::block_quant::{
        BlockQuant, BlockQuantFact, BlockQuantValue, Q4_0,
    };
    use tract_gpu::tensor::IntoDevice;

    pub(crate) fn run_mmm_test_case<K: GemmKernel>(
        (a_batch, b_batch, m, k, n): (usize, usize, usize, usize, usize),
        transpose_a: bool,
        transpose_b: bool,
        a_dt: DatumType,
        b_dt: DatumType,
    ) -> TractResult<()> {
        CUDA_STREAM.with(|stream| {
            let a_shape = if !transpose_a { [a_batch, m, k] } else { [a_batch, k, m] };
            let b_shape = if !transpose_b { [b_batch, k, n] } else { [b_batch, n, k] };
            let mut a = if a_dt == DatumType::F16 {
                Tensor::from_shape(
                    &a_shape,
                    &(0..a_batch * m * k)
                        .map(|f| f16::from_f32(f as f32 / (a_batch * m * k) as f32))
                        .collect::<Vec<_>>(),
                )?
            } else {
                Tensor::from_shape(
                    &a_shape,
                    &(0..a_batch * m * k)
                        .map(|f| f as f32 / (a_batch * m * k) as f32)
                        .collect::<Vec<_>>(),
                )?
            };

            let mut b = if b_dt == DatumType::F16 {
                Tensor::from_shape(
                    &b_shape,
                    &(0..b_batch * k * n)
                        .map(|f| f16::from_f32(f as f32 / (b_batch * n * k) as f32))
                        .collect::<Vec<_>>(),
                )?
            } else {
                Tensor::from_shape(
                    &b_shape,
                    &(0..b_batch * k * n)
                        .map(|f| f as f32 / (b_batch * m * k) as f32)
                        .collect::<Vec<_>>(),
                )?
            };

            let cuda_output = GemmImpl::<K>::new(transpose_a, transpose_b).eval(
                stream,
                &a.clone().into_device()?,
                &b.clone().into_device()?,
            )?;

            let matmul = PrefixMatMul {
                transpose_a,
                transpose_b,
                transpose_c: false,
                quantize_output: None,
            };

            // Compare to full precision
            if a_dt == DatumType::F16 && !(b_dt == DatumType::F16) {
                a = a.clone().cast_to_dt(DatumType::F32).unwrap().into_owned();
            }
            if b_dt == DatumType::F16 && !(a_dt == DatumType::F16) {
                b = b.clone().cast_to_dt(DatumType::F32).unwrap().into_owned();
            }

            let output = args_1!(matmul.eval(tvec![a.into_tvalue(), b.into_tvalue()])?);
            cuda_output.to_host()?.close_enough(&output, Approximation::VeryApproximate)?;
            Ok(())
        })
    }

    #[test]
    fn test_gemm_dispatches_params() -> TractResult<()> {
        let dt = DatumType::F32;
        let (m, k, n) = (2, 3, 4);
        assert_eq!(
            GemmDispatchParams::compute_dispatches_params::<GgmlGemm>(
                [dt; 3],
                &[1, m, k],
                false,
                &[1, k, n],
                false,
                false,
                &[1, m, n],
            )?,
            vec![GemmDispatchParams {
                dts: [dt; 3],
                a_batch: 1,
                b_batch: 1,
                m,
                n,
                k,
                transpose_a: false,
                a_offset: 0,
                transpose_b: false,
                b_offset: 0,
                q40_b: false,
                c_offset: 0,
                a_strides: natural_strides(&[1, m, k]),
                b_strides: natural_strides(&[1, k, n]),
                c_strides: natural_strides(&[1, m, n])
            }]
        );

        assert_eq!(
            GemmDispatchParams::compute_dispatches_params::<GgmlGemm>(
                [dt; 3],
                &[10, m, k],
                false,
                &[10, k, n],
                false,
                false,
                &[10, m, n],
            )?,
            vec![GemmDispatchParams {
                dts: [dt; 3],
                a_batch: 10,
                b_batch: 10,
                m,
                n,
                k,
                transpose_a: false,
                a_offset: 0,
                transpose_b: false,
                b_offset: 0,
                q40_b: false,
                c_offset: 0,
                a_strides: natural_strides(&[10, m, k]),
                b_strides: natural_strides(&[10, k, n]),
                c_strides: natural_strides(&[10, m, n])
            }]
        );

        assert_eq!(
            GemmDispatchParams::compute_dispatches_params::<GgmlGemm>(
                [dt; 3],
                &[1, m, k],
                false,
                &[2, k, n],
                false,
                false,
                &[2, m, n],
            )?,
            vec![
                GemmDispatchParams {
                    dts: [dt; 3],
                    a_batch: 1,
                    b_batch: 1,
                    m,
                    n,
                    k,
                    transpose_a: false,
                    a_offset: 0,
                    transpose_b: false,
                    b_offset: 0,
                    q40_b: false,
                    c_offset: 0,
                    a_strides: natural_strides(&[1, m, k]),
                    b_strides: natural_strides(&[2, k, n]),
                    c_strides: natural_strides(&[2, m, n])
                },
                GemmDispatchParams {
                    dts: [dt; 3],
                    a_batch: 1,
                    b_batch: 1,
                    m,
                    n,
                    k,
                    transpose_a: false,
                    a_offset: 0,
                    transpose_b: false,
                    b_offset: 1 * n * k * dt.size_of(),
                    q40_b: false,
                    c_offset: m * n * dt.size_of(),
                    a_strides: natural_strides(&[1, m, k]),
                    b_strides: natural_strides(&[2, k, n]),
                    c_strides: natural_strides(&[2, m, n])
                }
            ]
        );

        assert_eq!(
            GemmDispatchParams::compute_dispatches_params::<GgmlGemm>(
                [dt; 3],
                &[2, k, m],
                true,
                &[2, k, n],
                false,
                false,
                &[2, m, n],
            )?,
            vec![GemmDispatchParams {
                dts: [dt; 3],
                a_batch: 2,
                b_batch: 2,
                m,
                n,
                k,
                transpose_a: true,
                a_offset: 0,
                transpose_b: false,
                b_offset: 0,
                q40_b: false,
                c_offset: 0,
                a_strides: natural_strides(&[2, k, m]),
                b_strides: natural_strides(&[2, k, n]),
                c_strides: natural_strides(&[2, m, n])
            }]
        );

        assert_eq!(
            GemmDispatchParams::compute_dispatches_params::<GgmlGemm>(
                [dt; 3],
                &[2, k, m],
                true,
                &[1, k, n],
                false,
                false,
                &[2, m, n],
            )?,
            vec![
                GemmDispatchParams {
                    dts: [dt; 3],
                    a_batch: 1,
                    b_batch: 1,
                    m,
                    n,
                    k,
                    transpose_a: true,
                    a_offset: 0,
                    transpose_b: false,
                    b_offset: 0,
                    q40_b: false,
                    c_offset: 0,
                    a_strides: natural_strides(&[2, k, m]),
                    b_strides: natural_strides(&[1, k, n]),
                    c_strides: natural_strides(&[2, m, n])
                },
                GemmDispatchParams {
                    dts: [dt; 3],
                    a_batch: 1,
                    b_batch: 1,
                    m,
                    n,
                    k,
                    transpose_a: true,
                    a_offset: 1 * m * k * dt.size_of(),
                    transpose_b: false,
                    b_offset: 0,
                    q40_b: false,
                    c_offset: 1 * m * n * dt.size_of(),
                    a_strides: natural_strides(&[2, k, m]),
                    b_strides: natural_strides(&[1, k, n]),
                    c_strides: natural_strides(&[2, m, n])
                }
            ]
        );

        assert_eq!(
            GemmDispatchParams::compute_dispatches_params::<GgmlGemm>(
                [dt; 3],
                &[10, m, k],
                false,
                &[1, k, n],
                false,
                false,
                &[10, m, n],
            )?,
            vec![GemmDispatchParams {
                dts: [dt; 3],
                a_batch: 1,
                b_batch: 1,
                m: 10 * m,
                n,
                k,
                transpose_a: false,
                a_offset: 0,
                transpose_b: false,
                b_offset: 0,
                q40_b: false,
                c_offset: 0,
                a_strides: natural_strides(&[10, m, k]),
                b_strides: natural_strides(&[1, k, n]),
                c_strides: natural_strides(&[10, m, n])
            }]
        );

        Ok(())
    }

    #[test]
    fn test_squeeze_batch_axes() -> TractResult<()> {
        assert_eq!(squeeze_batch_axes(&[1, 2, 3, 4])?, tvec![2, 3, 4]);
        assert_eq!(squeeze_batch_axes(&[3, 2, 3, 4])?, tvec![6, 3, 4]);
        assert_eq!(squeeze_batch_axes(&[3, 1, 2, 3, 4])?, tvec![6, 3, 4]);
        assert!(squeeze_batch_axes(&[1]).is_err());
        assert_eq!(squeeze_batch_axes(&[1, 1, 3, 4])?, tvec![1, 3, 4]);
        Ok(())
    }

    proptest::proptest! {
        #[test]
        fn mmm_ggml_prop_f32(pb in <MmmProblem<GgmlGemm, f32>>::arbitrary_with(
            MmmProblemParams {
                force_k_as_inner_axis: true,
                q4_0_weights: false,
            }
        )) {
            let output = pb.run().unwrap();
            prop_assert!(output.close_enough(&pb.reference().unwrap(), Approximation::Approximate).is_ok())
        }

        #[test]
        fn mmm_ggml_prop_f16(pb in <MmmProblem<GgmlGemm, f16>>::arbitrary_with(
            MmmProblemParams {
                force_k_as_inner_axis: true,
                q4_0_weights: false,
            }
        )) {
            let output = pb.run().unwrap();
            prop_assert!(output.close_enough(&pb.reference().unwrap(), Approximation::VeryApproximate).is_ok())
        }

        #[test]
        fn mmm_ggml_prop_q4(pb in <MmmProblem<GgmlGemm, f32>>::arbitrary_with(
            MmmProblemParams {
                force_k_as_inner_axis: true,
                q4_0_weights: true,
            }
        )) {
            let output = pb.run().unwrap();
            prop_assert!(output.close_enough(&pb.reference().unwrap(), Approximation::VeryApproximate).is_ok())
        }
    }

    #[derive(Default, Debug, Clone)]
    pub struct MmmProblemParams {
        pub force_k_as_inner_axis: bool,
        pub q4_0_weights: bool,
    }

    #[derive(Debug)]
    pub struct MmmProblem<K: GemmKernel, F: Datum + Float>
    where
        F: Datum + Float,
        f32: AsPrimitive<F>,
    {
        pub b: usize,
        pub m: usize,
        pub k: usize,
        pub n: usize,
        pub lhs: Vec<F>,
        pub transpose_lhs: bool,
        pub rhs: Vec<F>,
        pub transpose_rhs: bool,
        pub q4_0: bool,
        pub _phantom: std::marker::PhantomData<K>,
    }

    impl<K, F> Arbitrary for MmmProblem<K, F>
    where
        K: GemmKernel,
        F: Datum + Float,
        f32: AsPrimitive<F>,
    {
        type Parameters = MmmProblemParams;
        type Strategy = BoxedStrategy<Self>;

        fn arbitrary_with(params: MmmProblemParams) -> Self::Strategy {
            (1usize..4, 1usize..16, 1usize..128, 1usize..16)
                .prop_flat_map(move |(b, m, mut k, n)| {
                    if params.q4_0_weights {
                        k = k.div_ceil(32) * 32
                    };

                    let lhs_len = b * m * k;
                    let rhs_len = b * n * k;
                    let datum = (0f32..1f32).prop_map(|x| x.as_());
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
                .prop_map(move |(b, m, k, n, lhs, mut transpose_lhs, rhs, mut transpose_rhs)| {
                    if params.force_k_as_inner_axis {
                        (transpose_lhs, transpose_rhs) = (false, true);
                    }
                    Self {
                        b,
                        m,
                        k,
                        n,
                        lhs,
                        transpose_lhs,
                        rhs,
                        transpose_rhs,
                        q4_0: params.q4_0_weights,
                        _phantom: std::marker::PhantomData,
                    }
                })
                .boxed()
        }
    }

    impl<K, F> MmmProblem<K, F>
    where
        K: GemmKernel,
        F: Datum + Float + std::ops::AddAssign,
        f32: AsPrimitive<F>,
    {
        pub fn reference(&self) -> TractResult<Tensor> {
            let matmul = PrefixMatMul {
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
            let mut rhs_tensor = if self.transpose_rhs {
                Tensor::from_shape(&[self.b, self.n, self.k], &self.rhs)?
            } else {
                Tensor::from_shape(&[self.b, self.k, self.n], &self.rhs)?
            };

            if self.q4_0 {
                rhs_tensor = Q4_0.simulate_precision_loss(rhs_tensor, 2)?
            };
            let output = matmul.eval(tvec![lhs_tensor.into_tvalue(), rhs_tensor.into_tvalue()])?;

            Ok(output[0].clone().into_tensor())
        }

        pub fn run(&self) -> TractResult<Tensor> {
            CUDA_STREAM.with(|stream| {
                let lhs = if self.transpose_lhs {
                    Tensor::from_shape(&[self.b, self.k, self.m], &self.lhs)?.into_device()?
                } else {
                    Tensor::from_shape(&[self.b, self.m, self.k], &self.lhs)?.into_device()?
                };
                let rhs = if self.transpose_rhs {
                    if !self.q4_0 {
                        Tensor::from_shape(&[self.b, self.n, self.k], &self.rhs)?
                    } else {
                        let b_quant = Q4_0.quant_f32(
                            &self
                                .rhs
                                .clone()
                                .into_iter()
                                .map(|x| x.to_f32().unwrap())
                                .collect_vec(),
                        )?;
                        let bqv = BlockQuantValue {
                            fact: BlockQuantFact::new(
                                Box::new(Q4_0),
                                tvec![self.b, self.n, self.k],
                            ),
                            value: Arc::new(b_quant),
                        };
                        let padded_q40 = pad_q40(&bqv)?;
                        tensor0(Opaque(Arc::new(padded_q40)))
                    }
                } else {
                    Tensor::from_shape(&[self.b, self.k, self.n], &self.rhs)?
                }
                .into_device()?;

                let matmul = GemmImpl::<K>::new(self.transpose_lhs, self.transpose_rhs);

                let c = matmul.eval(stream, &lhs, &rhs)?;
                Ok(c.to_host()?.into_tensor())
            })
        }
    }
}
