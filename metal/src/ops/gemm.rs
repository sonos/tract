use crate::kernels::mfa_gemm;
use anyhow::{bail, ensure};
use num_traits::Float;
use tract_core::broadcast;
use tract_core::internal::*;
use tract_core::ndarray;
use tract_core::ndarray::Dimension;

#[derive(Debug, Default, Clone)]
pub struct MetalGemm {}

impl Op for MetalGemm {
    fn name(&self) -> Cow<str> {
        "MetalGemm".into()
    }

    op_as_typed_op!();
}

impl MetalGemm {
    fn output_shape<D: DimLike>(&self, a: &[D], b: &[D]) -> TractResult<TVec<D>> {
        ensure!(a.len() == b.len());
        let a_rank = a.len();
        let b_rank = b.len();
        let m = a[a_rank - 2].clone();
        let n = b[b_rank - 1].clone();
        let mut c_shape = broadcast::multi_broadcast(&[&a[..a_rank - 2], &b[..b_rank - 2]])
            .context("Unable to broadcast")?;
        c_shape.push(m);
        c_shape.push(n);
        Ok(c_shape)
    }

    fn _eval<F: Datum + Float>(&self, a: TValue, b: TValue) -> TractResult<TVec<TValue>> {
        objc::rc::autoreleasepool(|| {
            crate::METAL_CONTEXT.with_borrow(|context| {
                let a_ptr = a.as_ptr::<F>()?;
                let b_ptr = b.as_ptr::<F>()?;
                let c_shape = self.output_shape(a.shape(), b.shape())?;
                let rank = c_shape.len();
                let m = c_shape[rank - 2];
                let n = c_shape[rank - 1];
                let k = a.shape()[rank - 1];

                let a_mk_strides = natural_strides(&[1, m, k]);
                let b_kn_strides = natural_strides(&[1, k, n]);
                unsafe {
                    let mut c = Tensor::uninitialized::<F>(&c_shape)?;
                    let c_ptr = c.as_ptr_mut::<F>()?;
                    let silent_a_axis = c.rank() - a.rank();
                    let silent_b_axis = c.rank() - b.rank();
                    for prefix in ndarray::indices(&c_shape[0..rank - 2]) {
                        let mut a_ptr = a_ptr;
                        let mut b_ptr = b_ptr;
                        let mut c_ptr = c_ptr;
                        for (axis, x) in prefix.as_array_view().iter().enumerate() {
                            if axis >= silent_a_axis && a.shape()[axis - silent_a_axis] != 1 {
                                a_ptr =
                                    a_ptr.offset(*x as isize * a.strides()[axis - silent_a_axis]);
                            }
                            if axis >= silent_b_axis && b.shape()[axis - silent_b_axis] != 1 {
                                b_ptr =
                                    b_ptr.offset(*x as isize * b.strides()[axis - silent_b_axis]);
                            }
                            c_ptr = c_ptr.offset(*x as isize * c.strides()[axis]);
                        }

                        mfa_gemm::mfa_gemm_with_slice(
                            context,
                            (1, m, n, k),
                            std::slice::from_raw_parts(a_ptr, m * k),
                            &a_mk_strides,
                            std::slice::from_raw_parts(b_ptr, k * n),
                            &b_kn_strides,
                            std::slice::from_raw_parts_mut(c_ptr, m * n),
                        )?;
                    }

                    Ok(tvec!(c.into_tvalue()))
                }
            })
        })
    }
}

impl EvalOp for MetalGemm {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let (a, b) = args_2!(inputs);
        if a.datum_type() == DatumType::F32 {
            self._eval::<f32>(a, b)
        } else if a.datum_type() == DatumType::F16 {
            self._eval::<f16>(a, b)
        } else {
            bail!("MetalGemm doesn't support this datum type: {:?}", a.datum_type())
        }
    }
}

impl TypedOp for MetalGemm {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        if inputs[0].datum_type == f16::datum_type() {
            ensure!(inputs[1].datum_type == f16::datum_type());
            Ok(tvec!(f16::fact(&self.output_shape(&inputs[0].shape, &inputs[1].shape)?)))
        } else {
            ensure!(inputs[0].datum_type == f32::datum_type());
            ensure!(inputs[1].datum_type == f32::datum_type());
            Ok(tvec!(f32::fact(&self.output_shape(&inputs[0].shape, &inputs[1].shape)?)))
        }
    }

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        let fma = self.output_shape(&inputs[0].shape, &inputs[1].shape)?.iter().product::<TDim>()
            * inputs[0].shape.last().unwrap();
        if inputs[0].datum_type == f16::datum_type() {
            Ok(tvec!((Cost::FMA(f16::datum_type()), fma)))
        } else {
            Ok(tvec!((Cost::FMA(f32::datum_type()), fma)))
        }
    }

    as_op!();
}
