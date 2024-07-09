use crate::kernels::matmul;
use crate::tensor::MetalTensorExt;
use crate::{IntoMetal, MetalContext};
use anyhow::{bail, ensure};
use num_traits::{Float, One};
use tract_core::internal::*;
use tract_core::ndarray;
use tract_core::ndarray::Dimension;

#[derive(Debug, Default, Clone)]
pub struct MetalGemm {
    pub transpose_a: bool,
    pub transpose_b: bool,
}

impl Op for MetalGemm {
    fn name(&self) -> Cow<str> {
        "MetalGemm".into()
    }

    op_as_typed_op!();
}

impl MetalGemm {
    fn output_shape<D: DimLike + One>(&self, a: &[D], b: &[D]) -> TVec<D> {
        let rank = a.len();
        let mut output: TVec<D> = (0..rank - 2)
            .map(|ix| if a[ix].is_one() { b[ix].clone() } else { a[ix].clone() })
            .collect();
        output.push(a[rank - 2 + self.transpose_a as usize].clone());
        output.push(b[rank - 2 + !self.transpose_b as usize].clone());
        output
    }

    fn resolve_output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let [a, b] = inputs else {
            bail!("Expects 2 inputs");
        };
        ensure!(a.rank() == b.rank());
        ensure!(a.rank() >= 2);
        ensure!(
            a.shape[a.rank() - 2 + !self.transpose_a as usize]
                == b.shape[b.rank() - 2 + self.transpose_b as usize]
        );

        if a.datum_type == f16::datum_type() {
            ensure!(b.datum_type == f16::datum_type());
            Ok(tvec!(f16::fact(&self.output_shape(&a.shape, &b.shape))))
        } else {
            ensure!(a.datum_type == f32::datum_type());
            ensure!(b.datum_type == f32::datum_type());
            Ok(tvec!(f32::fact(&self.output_shape(&a.shape, &b.shape))))
        }
    }

    fn dispatch_eval<F: Datum + Float>(
        &self,
        context: &MetalContext,
        a: &Tensor,
        b: &Tensor,
    ) -> TractResult<Tensor> {
        let a_ptr = a.as_ptr::<F>()?;
        let b_ptr = b.as_ptr::<F>()?;
        let c_shape = self.output_shape(a.shape(), b.shape());
        let rank = c_shape.len();
        let m = c_shape[rank - 2];
        let n = c_shape[rank - 1];
        let k = a.shape()[a.rank() - 2 + !self.transpose_a as usize];

        let a_strides = if self.transpose_a {
            natural_strides(&[1, k, m])
        } else {
            natural_strides(&[1, m, k])
        };
        let b_strides = if self.transpose_b {
            natural_strides(&[1, n, k])
        } else {
            natural_strides(&[1, k, n])
        };
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
                        a_ptr = a_ptr.offset(*x as isize * a.strides()[axis - silent_a_axis]);
                    }
                    if axis >= silent_b_axis && b.shape()[axis - silent_b_axis] != 1 {
                        b_ptr = b_ptr.offset(*x as isize * b.strides()[axis - silent_b_axis]);
                    }
                    c_ptr = c_ptr.offset(*x as isize * c.strides()[axis]);
                }

                matmul::mfa_dispatch_gemm_with_slice(
                    context,
                    (1, m, n, k),
                    std::slice::from_raw_parts(a_ptr, m * k),
                    &a_strides,
                    self.transpose_a,
                    std::slice::from_raw_parts(b_ptr, k * n),
                    &b_strides,
                    self.transpose_b,
                    std::slice::from_raw_parts_mut(c_ptr, m * n),
                )
                .with_context(|| {
                    anyhow!(
                        "Error while performing MatMul (a: {:?}), (b: {:?}) = (c: {:?})",
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

impl EvalOp for MetalGemm {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let (a, b) = args_2!(inputs);
        objc::rc::autoreleasepool(|| {
            crate::METAL_CONTEXT.with_borrow(|context| {
                if a.datum_type() == DatumType::F32 {
                    let out = self.dispatch_eval::<f32>(context, &a, &b)?.into_tvalue();
                    context.wait_until_completed()?;
                    Ok(tvec![out])
                } else if a.datum_type() == DatumType::F16 {
                    let out = self.dispatch_eval::<f16>(context, &a, &b)?.into_tvalue();
                    context.wait_until_completed()?;
                    Ok(tvec![out])
                } else if a.datum_type() == DatumType::Opaque {
                    let a_metal_ref = a.to_metal_tensor()?;
                    let b_metal_ref = b.to_metal_tensor()?;
                    let out = if a_metal_ref.datum_type() == DatumType::F16 {
                        self.dispatch_eval::<f16>(
                            context,
                            a_metal_ref.tensor(),
                            b_metal_ref.tensor(),
                        )?
                    } else if a_metal_ref.datum_type() == DatumType::F32 {
                        self.dispatch_eval::<f32>(
                            context,
                            a_metal_ref.tensor(),
                            b_metal_ref.tensor(),
                        )?
                    } else {
                        bail!(
                            "{:?} doesn't support this datum type: {:?} inside Opaque Metal tensor",
                            self.name(),
                            a_metal_ref.datum_type()
                        )
                    };
                    // We convert the output tensor as a metal tensor. Indeed the tensor will be accessible
                    // only when the command buffer will be commit and completed.
                    Ok(tvec![out.into_metal()?.into_opaque_tensor().into_tvalue()])
                } else {
                    bail!("{:?} doesn't support this datum type: {:?}", self.name(), a.datum_type())
                }
            })
        })
    }
}

impl TypedOp for MetalGemm {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::metal_output_facts(inputs, |input_facts| {
            self.resolve_output_facts(input_facts)
        })
        .with_context(|| {
            anyhow::anyhow!("Error while computing output facts for {:?}", self.name())
        })
    }

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        crate::utils::metal_facts(inputs, |input_facts| {
            let fma = self
                .output_shape(&input_facts[0].shape, &input_facts[1].shape)
                .iter()
                .product::<TDim>()
                * input_facts[0].shape.last().unwrap();
            if input_facts[0].datum_type == f16::datum_type() {
                Ok(tvec!((Cost::FMA(f16::datum_type()), fma)))
            } else {
                Ok(tvec!((Cost::FMA(f32::datum_type()), fma)))
            }
        })
        .with_context(|| anyhow::anyhow!("Error while computing cost for {:?}", self.name()))
    }

    as_op!();
}
