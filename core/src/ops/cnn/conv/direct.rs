use crate::internal::*;
use crate::ops::cnn::Patch;
use crate::ops::nn::DataShape;
use ndarray::prelude::*;
use num_traits::Zero;
use std::ops::{Add, AddAssign, Mul};
use tract_linalg::Conv;

#[derive(CustomDebug, Clone)]
pub struct Direct<T: Datum + Copy + Add + AddAssign + Mul + Zero + FloatLike> {
    conv: Box<Conv<T>>,
    patch: Patch,
    input_shape: DataShape,
    output_shape: DataShape,
    filters_as_group_o_i_hw: Option<Array4<T>>,
    packed_filters: Vec<Tensor>,
    ci_per_group: usize,
    co_per_group: usize,
    bias: Option<ArrayD<T>>,
}

impl<T> Direct<T>
where
    T: Datum + Copy + Add + AddAssign + Mul<T, Output = T> + Zero + FloatLike,
{
    pub fn new(
        input_shape: DataShape,
        patch: Patch,
        filters_as_group_o_i_hw: Array4<T>,
        bias: Option<ArrayD<T>>,
    ) -> TractResult<Direct<T>> {
        let group = filters_as_group_o_i_hw.shape()[0];
        let co_per_group = filters_as_group_o_i_hw.shape()[1];
        let ci_per_group = filters_as_group_o_i_hw.shape()[2];
        let output_shape = input_shape.fmt.from_n_c_hw(
            input_shape.n(),
            co_per_group * group,
            &*patch.output_shape,
        );
        let data_offsets: Vec<isize> = patch.centers_offsets();
        let channel_stride = input_shape.c_stride();
        let kernel_offsets: Vec<isize> = (0..ci_per_group)
            .flat_map(|c| {
                patch
                    .standard_layout_data_field
                    .iter()
                    .map(move |x| x + (c * channel_stride) as isize)
            })
            .collect();
        let conv = T::packed_direct_conv(co_per_group, kernel_offsets, data_offsets);

        let packed_filters = filters_as_group_o_i_hw
            .outer_iter()
            .map(|group| {
                let mut filter = unsafe {
                    Tensor::uninitialized_aligned::<T>(
                        &[conv.packed_a_len()],
                        conv.packed_a_alignment(),
                    )?
                };
                conv.pack_a(filter.as_ptr_mut()?, group.as_ptr(), group.strides()[0], 1);
                Ok(filter.into())
            })
            .collect::<TractResult<Vec<Tensor>>>()?;

        Ok(Direct {
            conv,
            patch,
            input_shape,
            output_shape,
            //filters_as_group_o_i_hw: if !patch.valid { Some(filters_as_group_o_i_hw) } else { None },
            filters_as_group_o_i_hw: Some(filters_as_group_o_i_hw),
            packed_filters,
            ci_per_group,
            co_per_group,
            bias,
        })
    }
}

impl<T> Op for Direct<T>
where
    T: Datum + Copy + Add + AddAssign + Mul<T, Output = T> + Zero + FloatLike,
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
where
    T: Datum + Copy + Add + AddAssign + Mul<T, Output = T> + Zero + FloatLike,
{
    fn eval(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let input = args_1!(inputs);
        unsafe {
            let input = input.to_array_view::<T>()?;
            let mut output = ArrayD::<T>::uninitialized(&*self.output_shape.shape);
            for n in 0..self.input_shape.n() {
                for g in 0..self.packed_filters.len() {
                    let input = input.slice_axis(Axis(0), (n..=n).into());
                    let mut output = output.slice_axis_mut(Axis(0), (n..=n).into());
                    let iptr = input
                        .as_ptr()
                        .offset((g * self.ci_per_group * self.input_shape.c_stride()) as isize);
                    let optr = output
                        .as_mut_ptr()
                        .offset((g * self.co_per_group * self.output_shape.c_stride()) as isize);
                    let filters =
                        self.filters_as_group_o_i_hw.as_ref().unwrap().index_axis(Axis(0), g);
                    self.patch.visit_output_by_zone(|pt| {
                        for co in 0..self.co_per_group {
                            let mut sum = T::zero();
                            let filter = filters.index_axis(Axis(0), co);
                            for ci in 0..self.ci_per_group {
                                let filter = filter.index_axis(Axis(0), ci);
                                let iptr = iptr.offset((self.input_shape.c_stride() * ci) as isize);
                                for (ix, offset) in pt.valid_offsets_with_indexes() {
                                    sum += filter.as_slice().unwrap()[ix]
                                        * *iptr.offset(offset as isize);
                                }
                            }
                            *optr.offset(
                                pt.output_offset() + (self.output_shape.c_stride() * co) as isize,
                            ) = sum;
                        }
                    });
                    /*
                    self.conv.conv(
                        self.packed_filters.get_unchecked(g).as_ptr()?,
                        input
                            .as_ptr()
                            .offset((g * self.ci_per_group * self.input_shape.c_stride()) as isize),
                        output.as_mut_ptr().offset(
                            (g * self.co_per_group * self.output_shape.c_stride()) as isize,
                        ),
                        self.output_shape.c_stride() as isize,
                        self.output_shape.w_stride() as isize,
                    );
                    */
                }
            }
            if let Some(ref bias) = self.bias {
                output += &bias;
            }
            Ok(tvec!(output.into()))
        }
    }
}

impl<T> InferenceRulesOp for Direct<T>
where
    T: Datum + Copy + Add + AddAssign + Mul + Zero + FloatLike,
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
