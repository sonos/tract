use crate::internal::*;
use crate::ops::nn::DataShape;
use crate::ops::cnn::Patch;
use ndarray::prelude::*;
use tract_linalg::Conv;
use num_traits::Zero;
use std::ops::{Add, Mul};

#[derive(CustomDebug, Clone)]
pub struct Direct<T: Datum + Copy + Add + Mul + Zero + FloatLike> {
    conv: Box<Conv<T>>,
    input_shape: DataShape,
    output_shape: DataShape,
    #[debug(skip)]
    packed_filters: Tensor,
}

impl<T> Direct<T>
where T: Datum + Copy + Add + Mul + Zero + FloatLike
{
    pub fn new(input_shape: DataShape, patch: Patch, filters_as_group_o_i_hw: ArrayView4<T>) -> TractResult<Direct<T>> {
        let output_shape = input_shape.fmt.from_n_c_hw(
            input_shape.n(),
            filters_as_group_o_i_hw.shape()[1],
            &*patch.output_shape,
        );
        let data_offsets: Vec<isize> = patch.centers_offsets();
        let channel_stride = input_shape.c_stride();
        let kernel_offsets: Vec<isize> = (0..input_shape.c())
            .flat_map(|c| {
                patch
                    .standard_layout_data_field
                    .iter()
                    .map(move |x| x + (c * channel_stride) as isize)
            })
            .collect();
        let conv = T::packed_direct_conv(output_shape.c(), kernel_offsets, data_offsets);

        let mut packed_filters = unsafe {
            Tensor::uninitialized_aligned::<T>(&[conv.packed_a_len()], conv.packed_a_alignment())?
        };
        conv.pack_a(
            packed_filters.as_slice_mut()?.as_mut_ptr(),
            filters_as_group_o_i_hw.as_slice().unwrap().as_ptr(),
            filters_as_group_o_i_hw.strides()[1],
            1
        );

        Ok(Direct { conv, input_shape, output_shape, packed_filters })
    }
}

impl<T> Op for Direct<T>
where T: Datum + Copy + Add + Mul + Zero + FloatLike
{
    fn name(&self) -> Cow<str> {
        format!("Conv::Direct<{:?}>", T::datum_type()).into()
    }

    fn info(&self) -> TractResult<Option<String>> {
        Ok(Some(format!("{:?}", self.conv)))
    }

    fn cost(&self, inputs: &[&TypedTensorInfo]) -> TractResult<TVec<(Cost, TDim)>> {
        let batch = inputs[0].shape.dim(0);
        Ok(tvec!((
            Cost::FMA(T::datum_type()),
            batch * self.conv.n() * self.conv.co() * self.conv.k()
        )))
    }

    fn rounding_errors(&self) -> bool {
        true
    }
}

impl<T> StatelessOp for Direct<T>
where T: Datum + Copy + Add + Mul + Zero + FloatLike
{
    fn eval(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let input = args_1!(inputs);
        unsafe {
            let input = input.to_array_view::<T>()?;
            let mut output = ArrayD::<T>::uninitialized(&*self.output_shape.shape);
            for n in 0..self.input_shape.n() {
                let input = input.slice_axis(Axis(0), (n..=n).into());
                let mut output = output.slice_axis_mut(Axis(0), (n..=n).into());
                self.conv.conv(
                    self.packed_filters.as_slice::<T>()?.as_ptr(),
                    input.as_ptr(),
                    output.as_mut_ptr(),
                    self.output_shape.c_stride() as isize,
                    self.output_shape.w_stride() as isize,
                );
            }
            Ok(tvec!(output.into()))
        }
    }
}

impl<T> InferenceRulesOp for Direct<T>
where T: Datum + Copy + Add + Mul + Zero + FloatLike
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
