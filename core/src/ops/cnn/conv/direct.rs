use crate::internal::*;
use crate::ops::cnn::Patch;
use crate::ops::nn::DataShape;
use ndarray::prelude::*;
use num_traits::Zero;
use std::ops::{Add, AddAssign, Mul};
use tract_linalg::Conv;

#[derive(CustomDebug, Clone)]
pub struct Direct<T: Datum + Copy + Add + AddAssign + Mul + Zero + FloatLike> {
    conv_direct: Option<(Box<Conv<T>>, Vec<Tensor>, isize)>,
    patch: Patch,
    input_shape: DataShape,
    output_shape: DataShape,
    filters_as_group_o_i_hw: Option<Array4<T>>,
    ci_per_group: usize,
    co_per_group: usize,
    group: usize,
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
        let channel_stride = input_shape.c_stride();

        let conv_direct = if let Some(zone_id) = patch.valid_zone {
            let zone = &patch.zones[zone_id];
            let kernel_offsets: Vec<isize> = (0..ci_per_group)
                .flat_map(|c| {
                    patch
                        .standard_layout_data_field
                        .iter()
                        .map(move |x| x + (c * channel_stride) as isize)
                })
                .collect();

            // translate the valid zone into the overlapping semi-valid (but safe) zone
            let data_offsets = patch.centers_offsets();

            let first_of_valid_zone = zone
                .output_ranges()
                .iter()
                .zip(&patch.op_strides_times_input_storage_strides)
                .map(|(a, b)| a.start as isize * *b)
                .sum::<isize>();
            let first_of_valid_zone_ix = data_offsets.iter().position(|&a| a >= first_of_valid_zone).unwrap();
            let last_of_valid_zone = zone
                .output_ranges()
                .iter()
                .zip(&patch.op_strides_times_input_storage_strides)
                .map(|(a, b)| (a.end - 1) as isize * *b)
                .sum::<isize>();
            let first_of_invalid_zone_ix = data_offsets.iter().rposition(|&a| a <= last_of_valid_zone).unwrap() + 1;
            let pseudo_valid_range = patch.centers_offsets()[first_of_valid_zone_ix..first_of_invalid_zone_ix].to_vec();

            let conv = T::packed_direct_conv(co_per_group, kernel_offsets, pseudo_valid_range);

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

            let output_offset = zone
                .output_ranges()
                .iter()
                .zip(&patch.output_layout_strides)
                .map(|(r, s)| r.start as isize * s)
                .sum::<isize>();

            Some((conv, packed_filters, output_offset))
        } else {
            None
        };

        Ok(Direct {
            conv_direct,
            patch,
            input_shape,
            output_shape,
            //filters_as_group_o_i_hw: if !patch.valid { Some(filters_as_group_o_i_hw) } else { None },
            filters_as_group_o_i_hw: Some(filters_as_group_o_i_hw),
            ci_per_group,
            co_per_group,
            group,
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
        Ok(Some(format!("{:?}", self.conv_direct)))
    }

    fn cost(&self, inputs: &[&TypedTensorInfo]) -> TractResult<TVec<(Cost, TDim)>> {
        let batch = inputs[0].shape.dim(0);
        Ok(tvec!((
            Cost::FMA(T::datum_type()),
            batch
                * self.group
                * self.co_per_group
                * self.ci_per_group
                * self.patch.spec.kernel_shape.iter().cloned().product::<usize>()
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
                for g in 0..self.group {
                    let input = input.slice_axis(Axis(0), (n..=n).into());
                    let mut output = output.slice_axis_mut(Axis(0), (n..=n).into());
                    let iptr = input
                        .as_ptr()
                        .offset((g * self.ci_per_group * self.input_shape.c_stride()) as isize);
                    let optr = output
                        .as_mut_ptr()
                        .offset((g * self.co_per_group * self.output_shape.c_stride()) as isize);
                    if let Some((ref cv, ref filters, output_offset)) = self.conv_direct {
                        cv.conv(
                            filters.get_unchecked(g).as_ptr()?,
                            iptr,
                            optr.offset(output_offset),
                            self.output_shape.c_stride() as isize,
                            self.output_shape.w_stride() as isize,
                        );
                    }
                    if let Some(ref filters) = self.filters_as_group_o_i_hw {
                        let fptr = filters.slice_axis(Axis(0), (g..=g).into()).as_ptr();
                        self.patch.visit_invalid(|pt| {
                            for co in 0..self.co_per_group {
                                let fptr = fptr.offset(co as isize * filters.strides().get_unchecked(1));
                                let mut sum = T::zero();
                                for ci in 0..self.ci_per_group {
                                    let fptr = fptr.offset(ci as isize * filters.strides().get_unchecked(2));
                                    let iptr = iptr.offset((self.input_shape.c_stride() * ci) as isize);
                                    for (ix, offset) in pt.valid_offsets_with_indexes() {
                                        sum += *fptr.offset(ix as isize) * *iptr.offset(offset as isize);
                                    }
                                }
                                *optr.offset(
                                    pt.output_offset() + (self.output_shape.c_stride() * co) as isize,
                                ) = sum;
                            }
                        });
                    }
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
