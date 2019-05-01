use crate::internal::*;
use crate::ops::cnn::Patch;
use crate::ops::nn::DataShape;
use ndarray::*;
use std::iter::Sum;

#[derive(Debug, Clone, new)]
pub struct DepthWise<T>
where
    T: Datum + Clone + ndarray::LinalgScalar + std::ops::AddAssign<T> + PartialEq + Sum,
{
    patch: Patch,
    input_shape: DataShape,
    output_shape: DataShape,
    kernel_chw: ArrayD<T>,
    bias: Option<ArrayD<T>>,
}

impl<T> DepthWise<T>
where
    T: Datum + Clone + ndarray::LinalgScalar + std::ops::AddAssign<T> + PartialEq + Sum,
{
    fn evald(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let img = args_1!(inputs);
        let img = img.to_array_view::<T>()?;
        let iptr = img.as_ptr();
        let mut output = unsafe { ArrayD::<T>::uninitialized(&*self.output_shape.shape) };
        let optr = output.as_mut_ptr();
        let k_stride_o = self.kernel_chw.strides()[0];
        let k_stride_i = self.kernel_chw.strides()[1];
        let mult = self.output_shape.c() / self.input_shape.c();
        unsafe {
            self.patch.visit_output_in_order(|visitor| {
                for n in 0..self.input_shape.n() {
                    let input_offset = self.input_shape.n_stride() * n;
                    let output_offset = self.output_shape.n_stride() * n;
                    for c in 0..self.input_shape.c() {
                        let input_offset = input_offset + self.input_shape.c_stride() * c;
                        for m in 0..mult {
                            let output_offset =
                                output_offset + self.output_shape.c_stride() * (m + c * mult);
                            let kptr =
                                self.kernel_chw.as_ptr().offset(k_stride_i * c as isize + k_stride_o * m as isize);
                            let mut sum = T::zero();
                            for (ix, v) in visitor.valid_offsets_with_indexes() {
                                let k = *kptr.offset(ix as isize);
                                let i = *iptr.offset(input_offset as isize + v);
                                sum += k * i;
                            }
                            *optr.offset(output_offset as isize + visitor.output_offset()) = sum;
                        }
                    }
                }
            });
        }
        if let Some(ref bias) = self.bias {
            output += bias;
        }
        Ok(tvec!(output.into()))
    }
}

impl<T> Op for DepthWise<T>
where
    T: Datum + Clone + ndarray::LinalgScalar + std::ops::AddAssign<T> + PartialEq + Sum,
{
    fn name(&self) -> Cow<str> {
        format!("Conv::DepthWise<{:?}>", T::datum_type()).into()
    }

    fn cost(&self, _inputs: &[&TypedTensorInfo]) -> TractResult<TVec<(Cost, TDim)>> {
        let n_output_points = self.patch.output_shape.iter().cloned().product::<usize>();
        Ok(tvec!((
            Cost::FMA(T::datum_type()),
            (self.input_shape.n() * n_output_points * self.kernel_chw.len()).to_dim()
        )))
    }
}

impl<T> StatelessOp for DepthWise<T>
where
    T: Datum + Clone + ndarray::LinalgScalar + std::ops::AddAssign<T> + PartialEq + Sum,
{
    fn eval(&self, inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        match inputs[0].shape().len() {
            /*
            3 => self.eval3(inputs),
            4 => self.eval4(inputs),
            */
            _ => self.evald(inputs),
        }
    }
}

impl<T> InferenceRulesOp for DepthWise<T>
where
    T: Datum + Clone + ndarray::LinalgScalar + std::ops::AddAssign<T> + PartialEq + Sum,
{
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        _s: &mut Solver<'r>,
        _inputs: &'p [TensorProxy],
        _outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        unreachable!()
    }
}
