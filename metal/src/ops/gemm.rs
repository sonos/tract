use crate::kernels::matmul;
use crate::tensor::MetalTensorExt;
use crate::{MetalContext, MetalTensor};
use anyhow::{bail, ensure};
use num_traits::One;
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

    fn dispatch_eval(
        &self,
        context: &MetalContext,
        a: &MetalTensor,
        b: &MetalTensor,
    ) -> TractResult<MetalTensor> {
        let c_dt = a.datum_type();
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
            let c = MetalTensor::uninitialized_dt(c_dt, &c_shape)?;
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

                matmul::dispatch_metal_mfa_gemm(
                    context,
                    matmul::GemmPrecision::from_dt(c_dt)?,
                    (1, m, n, k),
                    std::mem::transmute::<&[isize], &[usize]>(a_strides.as_slice()),
                    a_offset as usize * c_dt.size_of(),
                    &a.metal(),
                    self.transpose_a,
                    std::mem::transmute::<&[isize], &[usize]>(b_strides.as_slice()),
                    b_offset as usize * c_dt.size_of(),
                    &b.metal(),
                    self.transpose_b,
                    &c.metal(),
                    c_offset as usize * c_dt.size_of(),
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
                let a_metal_ref = a.to_metal_tensor()?;
                let b_metal_ref = b.to_metal_tensor()?;
                let out = self.dispatch_eval(context, a_metal_ref, b_metal_ref)?;
                Ok(tvec![out.into_opaque_tensor().into_tvalue()])
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
