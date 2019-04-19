use crate::internal::*;
use crate::ops::nn::Patch;
use ndarray::*;
use std::iter::Sum;

use unsafe_unwrap::UnsafeUnwrap;

#[derive(Debug, Clone, new)]
pub struct DepthWise<T>
where
    T: Datum + Clone + ndarray::LinalgScalar + std::ops::AddAssign<T> + PartialEq + Sum,
{
    patch: Patch,
    out_channels: usize,
    kernel_chw: ArrayD<T>,
    bias: Option<ArrayD<T>>,
}

impl<T> DepthWise<T>
where
    T: Datum + Clone + ndarray::LinalgScalar + std::ops::AddAssign<T> + PartialEq + Sum,
{
    fn eval3(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let img = args_1!(inputs);
        let img = img.to_array_view::<T>()?;
        let visitor = self.patch.wrap(&img);
        let output_shape = self.patch.output_full_shape(self.out_channels);
        let mut output = Array3::<T>::from_shape_fn(
            (output_shape[0], output_shape[1], output_shape[2]),
            |coords| {
                let coords = [coords.0, coords.1, coords.2];
                let it = visitor.at(&coords);
                let channel = coords[self.patch.input_shape.c_axis()];
                let kernel = self.kernel_chw.slice_axis(Axis(0), (channel..=channel).into());
                kernel.iter().zip(it).map(|(&k, v)| k * v.unwrap_or(T::zero())).sum()
            },
        );
        if let Some(ref bias) = self.bias {
            output += bias;
        }
        Ok(tvec!(output.into()))
    }

    fn eval4(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let img = args_1!(inputs);
        let img = img.to_array_view::<T>()?;
        let visitor = self.patch.wrap(&img);
        let output_shape = self.patch.output_full_shape(self.out_channels);
        let len = self.patch.kernel_spatial_shape.iter().cloned().product::<usize>();
        let c_axis = self.patch.input_shape.c_axis();
        let kernel = self.kernel_chw.as_slice_memory_order().unwrap();
        let kernel_stride: usize = self.kernel_chw.strides()[0] as usize;
        let mut output = Array4::<T>::from_shape_fn(
            (output_shape[0], output_shape[1], output_shape[2], output_shape[3]),
            |coords| {
                let coords = [coords.0, coords.1, coords.2, coords.3];
                let mut it = visitor.at(&coords);
                unsafe {
                    let channel = coords.get_unchecked(c_axis);
                    let k = kernel.get_unchecked((kernel_stride * channel)..);
                    let mut sum = T::zero();
                    for i in 0..len {
                        sum += *k.get_unchecked(i) * it.next().unsafe_unwrap().unwrap_or(T::zero())
                    }
                    sum
                }
            },
        );
        if let Some(ref bias) = self.bias {
            output += bias;
        }
        Ok(tvec!(output.into()))
    }

    fn evald(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let img = args_1!(inputs);
        let img = img.to_array_view::<T>()?;
        let visitor = self.patch.wrap(&img);
        let output_shape = self.patch.output_full_shape(self.out_channels);
        let mut output = ArrayD::<T>::from_shape_fn(&*output_shape, |coords| {
            let it = visitor.at(&coords.slice());
            let channel = coords[self.patch.input_shape.c_axis()];
            let kernel = self.kernel_chw.slice_axis(Axis(0), (channel..=channel).into());
            kernel.iter().zip(it).map(|(&k, v)| k * v.unwrap_or(T::zero())).sum()
        });
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
        let n_output_points = self.patch.output_spatial_shape.iter().cloned().product::<usize>();
        Ok(tvec!((
            Cost::FMA(T::datum_type()),
            (self.patch.input_shape.n() * n_output_points * self.kernel_chw.len()).to_dim()
        )))
    }
}

impl<T> StatelessOp for DepthWise<T>
where
    T: Datum + Clone + ndarray::LinalgScalar + std::ops::AddAssign<T> + PartialEq + Sum,
{
    fn eval(&self, inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        match inputs[0].shape().len() {
            3 => self.eval3(inputs),
            4 => self.eval4(inputs),
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
