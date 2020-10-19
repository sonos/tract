use crate::internal::*;
use crate::ops::cnn::Patch;
use crate::ops::nn::DataShape;
use ndarray::*;

#[derive(Debug, Clone, new, Hash)]
pub struct DepthWise {
    patch: Patch,
    input_shape: DataShape,
    output_shape: DataShape,
    kernel_chw: Arc<Tensor>,
    bias: Option<Arc<Tensor>>,
}

tract_data::impl_dyn_hash!(DepthWise);

impl Op for DepthWise {
    fn name(&self) -> Cow<str> {
        "DepthWiseConv".into()
    }

    op_core_lir!();
    op_as_typed_op!();
}

impl EvalOp for DepthWise {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        dispatch_floatlike!(Self::eval_t(inputs[0].datum_type())(self, inputs))
    }
}

impl DepthWise {
    fn eval_t<T: Datum + Copy + num_traits::Zero + ndarray::LinalgScalar>(
        &self,
        mut inputs: TVec<Arc<Tensor>>,
    ) -> TractResult<TVec<Arc<Tensor>>> {
        let img = args_1!(inputs);
        let img = img.to_array_view::<T>()?;
        let iptr = img.as_ptr();
        let mut output = unsafe { ArrayD::<T>::uninitialized(&*self.output_shape.shape) };
        let optr = output.as_mut_ptr();
        let kernel_chw = self.kernel_chw.to_array_view::<T>()?;
        let k_stride_o = kernel_chw.strides()[0];
        let k_stride_i = kernel_chw.strides()[1];
        let mult = *self.output_shape.c() / *self.input_shape.c();
        let n = *self.input_shape.n().unwrap_or(&1);
        let n_stride_i = *self.input_shape.n_stride().unwrap_or(&0);
        let n_stride_o = *self.output_shape.n_stride().unwrap_or(&0);
        let c_stride_i = *self.input_shape.c_stride();
        let c_stride_o = *self.output_shape.c_stride();
        let bias = self.bias.as_ref().map(|b| b.as_slice::<T>()).transpose()?;
        unsafe {
            self.patch.visit_output(|visitor| {
                for n in 0..n {
                    let input_offset = n_stride_i * n;
                    let output_offset = n_stride_o * n;
                    for c in 0..*self.input_shape.c() {
                        let input_offset = input_offset + c_stride_i * c;
                        for m in 0..mult {
                            let mut sum = if let Some(b) = &bias {
                                *b.get_unchecked(m + c * mult)
                            } else {
                                T::zero()
                            };
                            let output_offset = output_offset + c_stride_o * (m + c * mult);
                            let kptr = kernel_chw
                                .as_ptr()
                                .offset(k_stride_i * c as isize + k_stride_o * m as isize);
                            for (ix, v) in visitor.valid_offsets_with_indexes() {
                                let k = *kptr.offset(ix as isize);
                                let i = *iptr.offset(input_offset as isize + v);
                                sum = sum + k * i;
                            }
                            let ptr = optr.offset(output_offset as isize + visitor.output_offset);
                            *ptr = sum;
                        }
                    }
                }
            });
        }
        Ok(tvec!(output.into_arc_tensor()))
    }
}

impl TypedOp for DepthWise {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(TypedFact::dt_shape(inputs[0].datum_type, &*self.output_shape.shape)?))
    }

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        let n_output_points = self.patch.output_shape.iter().cloned().product::<usize>();
        Ok(tvec!((
            Cost::FMA(inputs[0].datum_type),
            (self.input_shape.n().unwrap_or(&1) * n_output_points * self.kernel_chw.len()).to_dim()
        )))
    }

    as_op!();
}
