use crate::internal::*;
use crate::ops::cnn::Patch;
use crate::ops::nn::DataShape;
use ndarray::*;
use std::iter::Sum;

use unsafe_unwrap::UnsafeUnwrap;

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
    /*
    fn eval3(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let img = args_1!(inputs);
        let img = img.to_array_view::<T>()?;
        let ptr = img.as_ptr();
        let visitor = self.patch.wrap(&img);
        let shape = &self.patch.input_shape;
        let output_shape = self.patch.output_full_shape(self.out_channels);
        let mut output = unsafe {
            Array3::<T>::from_shape_fn(
                (output_shape[0], output_shape[1], output_shape[2]),
                |coords| {
                    let coords = [coords.0, coords.1, coords.2];
                    let ptr = ptr.offset((shape.n_stride() * coords[shape.n_axis()]) as isize);
                    let ptr = ptr.offset((shape.c_stride() * coords[shape.c_axis()]) as isize);
                    let it = visitor.attt(&coords[shape.hw_axes()]);
                    let channel = coords[self.patch.input_shape.c_axis()];
                    let kernel = self.kernel_chw.slice_axis(Axis(0), (channel..=channel).into());
                    kernel
                        .iter()
                        .zip(it)
                        .map(|(&k, v)| k * v.map(|v| *ptr.offset(v)).unwrap_or(T::zero()))
                        .sum()
                },
            )
        };
        if let Some(ref bias) = self.bias {
            output += bias;
        }
        Ok(tvec!(output.into()))
    }

    fn eval4(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let img = args_1!(inputs);
        let img = img.to_array_view::<T>()?;
        let iptr = img.as_ptr();
        let len = self.patch.spec.kernel_shape.iter().cloned().product::<usize>();
        let kernel = self.kernel_chw.as_slice_memory_order().unwrap();
        let kernel_stride: usize = self.kernel_chw.strides()[0] as usize;
        let mut output = unsafe { ArrayD::<T>::uninitialized(&*self.output_shape.shape) };
        let optr = output.as_mut_ptr();
        for n in 0..self.input_shape.n() {
            unsafe {
                let iptr = iptr.offset((self.input_shape.n_stride() * n) as isize);
                let optr = optr.offset((self.output_shape.n_stride() * n) as isize);
                for (coords, hint) in self.patch.visit_all_2() {
                    let optr = optr.offset(
                        coords.0 as isize
                            * *self.output_shape.hw_strides().get_unchecked(0) as isize,
                    );
                    let optr = optr.offset(
                        coords.1 as isize
                            * *self.output_shape.hw_strides().get_unchecked(1) as isize,
                    );
                    for c in 0..self.output_shape.c() {
                        let iptr = iptr.offset(c as isize * self.input_shape.c_stride() as isize);
                        let optr = optr.offset(c as isize * self.output_shape.c_stride() as isize);
                        let k = kernel.get_unchecked((kernel_stride * c)..);
                        let mut sum = T::zero();
                        let mut it = self.patch.at_hint(&[coords.0, coords.1], hint);
                        for i in 0..len {
                            let vk = *k.get_unchecked(i);
                            let vi =
                                it.next().unwrap().map(|o| *iptr.offset(o)).unwrap_or(T::zero());
                            sum += vk * vi;
                        }
                        *optr = sum
                    }
                }
            }
        }
        if let Some(ref bias) = self.bias {
            output += bias;
        }
        Ok(tvec!(output.into()))
    }
    */

    fn evald(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let img = args_1!(inputs);
        let img = img.to_array_view::<T>()?;
        let ptr = img.as_ptr();
        let shape = &self.input_shape;
        let mut output = unsafe {
            ArrayD::<T>::from_shape_fn(&*self.output_shape.shape, |coords| {
                let ptr = ptr.offset((shape.n_stride() * coords[shape.n_axis()]) as isize);
                let ptr = ptr.offset((shape.c_stride() * coords[shape.c_axis()]) as isize);
                let it = self.patch.at(&coords.slice()[shape.hw_axes()]);
                let channel = coords[shape.c_axis()];
                let kernel = self.kernel_chw.slice_axis(Axis(0), (channel..=channel).into());
                kernel
                    .iter()
                    .zip(it)
                    .map(|(&k, v)| k * v.map(|o| *ptr.offset(o)).unwrap_or(T::zero()))
                    .sum()
            })
        };;
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
