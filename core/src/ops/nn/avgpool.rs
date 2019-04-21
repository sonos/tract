use crate::internal::*;
use ndarray::prelude::*;
use num_traits::{AsPrimitive, Float};

use super::{DataFormat, PaddingSpec, Patch};
use crate::ops::nn::patches::PatchVisitor;

#[derive(Debug, Clone, new, Default)]
pub struct AvgPool {
    data_fmt: DataFormat,
    kernel_shape: TVec<usize>,
    padding: PaddingSpec,
    strides: Option<TVec<usize>>,
    count_include_pad: bool,
}

impl AvgPool {
    fn patch(&self, input_full_shape: &[usize]) -> Patch {
        let hw_rank = self.data_fmt.shape(input_full_shape).hw_rank();
        Patch::new(
            self.data_fmt,
            tvec![1; hw_rank],
            self.kernel_shape.clone(),
            &self.padding,
            self.strides.clone().unwrap_or_else(|| tvec![1; hw_rank]),
            input_full_shape.into(),
        )
    }

    fn eval_t<T>(&self, inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>>
    where
        T: Datum + Float,
        usize: AsPrimitive<T>,
    {
        let patch = self.patch(inputs[0].shape());
        FixedAvgPool::new(patch, self.count_include_pad).eval(inputs)
    }
}

impl Op for AvgPool {
    fn name(&self) -> Cow<str> {
        "AvgPool".into()
    }

    fn codegen(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let inputs = model.node_input_facts(node.id)?;
        if let Some(shape) = inputs[0].shape.as_finite() {
            let dt = inputs[0].datum_type;
            let patch = self.patch(&*shape);
            fn fixed<T>(patch: Patch, count_include_pad: bool) -> Box<Op>
            where
                T: Datum + Float,
                usize: AsPrimitive<T>,
            {
                Box::new(FixedAvgPool::new(patch, count_include_pad))
            }
            let op = dispatch_floatlike!(fixed(dt)(patch, self.count_include_pad));
            return Ok(Some(TypedModelPatch::single_unary_op(model, node, op)?));
        }
        Ok(None)
    }
}

impl StatelessOp for AvgPool {
    fn eval(&self, inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        dispatch_floatlike!(Self::eval_t(inputs[0].datum_type())(self, inputs))
    }
}

impl InferenceRulesOp for AvgPool {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        s.equals(&outputs[0].rank, &inputs[0].rank)?;
        s.given(&inputs[0].shape, move |s, ishape| {
            let ishape = self.data_fmt.shape(ishape);
            let ones = tvec![1; ishape.hw_rank()];
            let computed = self.padding.compute(
                ishape.hw_dims(),
                &*self.kernel_shape,
                &ones,
                self.strides.as_ref().unwrap_or(&ones),
            );
            for (ix, d) in computed.iter().enumerate() {
                s.equals(&outputs[0].shape[ix + ishape.h_axis()], d.output)?;
            }
            s.equals(&outputs[0].shape[ishape.n_axis()], ishape.n_dim())?;
            s.equals(&outputs[0].shape[ishape.c_axis()], ishape.c_dim())?;
            Ok(())
        })
    }
}

#[derive(Debug, Clone, new)]
pub struct FixedAvgPool<T>
where
    T: Datum + Float,
    usize: AsPrimitive<T>,
{
    patch: Patch,
    count_include_pad: bool,
    _phantom: PhantomData<T>,
}

impl<T: Datum + Float> FixedAvgPool<T>
where
    T: Datum + Float,
    usize: AsPrimitive<T>,
{
    #[inline(never)]
    fn valid_2d_nhwc(&self, input: &ArrayView4<T>, output: &mut ArrayViewMut4<T>) {
        unsafe {
            let out_h = output.shape()[self.patch.input_shape.hw_axes()][0];
            let out_w = output.shape()[self.patch.input_shape.hw_axes()][1];
            let stride_in_y = input.strides()[self.patch.input_shape.hw_axes()][0]
                * self.patch.kernel_strides[0] as isize;
            let stride_in_x = input.strides()[self.patch.input_shape.hw_axes()][1]
                * self.patch.kernel_strides[1] as isize;
            let stride_out_y = output.strides()[self.patch.input_shape.hw_axes()][0];
            let stride_out_x = output.strides()[self.patch.input_shape.hw_axes()][1];
            let k_len = self.patch.kernel_spatial_shape.iter().cloned().product::<usize>().as_();
            for i in 0..self.patch.input_shape.n() {
                let p_in = input
                    .slice_axis(Axis(self.patch.input_shape.n_axis()), (i..=i).into())
                    .as_ptr();
                let p_out = output
                    .slice_axis_mut(Axis(self.patch.input_shape.n_axis()), (i..=i).into())
                    .as_mut_ptr();
                for y in 0..out_h {
                    let p_in = p_in.offset(stride_in_y * y as isize);
                    let p_out = p_out.offset(stride_out_y * y as isize);
                    for x in 0..out_w {
                        let p_in = p_in.offset(stride_in_x * x as isize);
                        let p_out = p_out.offset(stride_out_x * x as isize);
                        let max_c = self.patch.input_shape.c();
                        for c in 0..max_c / 8 {
                            let p_in = p_in.offset(8 * c as isize);
                            let mut v = [T::zero(); 8];
                            for k in &self.patch.standard_layout_data_field {
                                let input = std::slice::from_raw_parts(p_in.offset(*k), 8);
                                for (v, i) in v.iter_mut().zip(input.iter()) {
                                    *v = *v + *i;
                                }
                            }
                            for (ix, v) in v.iter_mut().enumerate() {
                                *p_out.offset((c * 8 + ix) as isize) = *v / k_len;
                            }
                        }
                        for c in 0..max_c % 8 {
                            let p_in = p_in.offset((max_c / 8 * 8 + c) as isize);
                            let mut v = T::zero();
                            for k in &self.patch.standard_layout_data_field {
                                v = v + *p_in.offset(*k);
                            }
                            let p_out = p_out.offset((max_c / 8 * 8 + c) as isize);
                            *p_out = v / k_len;
                        }
                    }
                }
            }
        }
    }

    #[inline(never)]
    fn valid_2d(&self, input: &ArrayView4<T>, output: &mut ArrayViewMut4<T>) {
        unsafe {
            let out_h = output.shape()[self.patch.input_shape.hw_axes()][0];
            let out_w = output.shape()[self.patch.input_shape.hw_axes()][1];
            let stride_in_c = input.strides()[self.patch.input_shape.c_axis()] as isize;
            let stride_in_y = input.strides()[self.patch.input_shape.hw_axes()][0]
                * self.patch.kernel_strides[0] as isize;
            let stride_in_x = input.strides()[self.patch.input_shape.hw_axes()][1]
                * self.patch.kernel_strides[1] as isize;
            let stride_out_c = output.strides()[self.patch.input_shape.c_axis()] as isize;
            let stride_out_y = output.strides()[self.patch.input_shape.hw_axes()][0];
            let stride_out_x = output.strides()[self.patch.input_shape.hw_axes()][1];
            let k_len = self.patch.kernel_spatial_shape.iter().cloned().product::<usize>().as_();
            for i in 0..self.patch.input_shape.n() {
                let p_in_i = input
                    .slice_axis(Axis(self.patch.input_shape.n_axis()), (i..=i).into())
                    .as_ptr();
                let p_out_i = output
                    .slice_axis_mut(Axis(self.patch.input_shape.n_axis()), (i..=i).into())
                    .as_mut_ptr();
                for c in 0..self.patch.input_shape.c() {
                    let p_in_ic = p_in_i.offset(c as isize * stride_in_c);
                    let p_out_ic = p_out_i.offset(c as isize * stride_out_c);
                    for y in 0..out_h {
                        let p_in_icy = p_in_ic.offset(stride_in_y * y as isize);
                        let p_out_icy = p_out_ic.offset(stride_out_y * y as isize);
                        for x in 0..out_w {
                            let p_in_icyx = p_in_icy.offset(stride_in_x * x as isize);
                            let p_out_icyx = p_out_icy.offset(stride_out_x * x as isize);
                            let mut v = T::zero();
                            for k in &self.patch.standard_layout_data_field {
                                v = v + *p_in_icyx.offset(*k);
                            }
                            *p_out_icyx = v / k_len;
                        }
                    }
                }
            }
        }
    }

    pub fn two_d(&self, input: &ArrayView4<T>) -> TractResult<Array4<T>> {
        let output_shape: TVec<usize> =
            self.patch.output_full_shape(self.patch.input_shape.c_dim());
        let mut output = unsafe { ArrayD::uninitialized(&*output_shape).into_dimensionality()? };
        if !self.patch.padded {
            if self.patch.input_shape.fmt == DataFormat::NHWC {
                self.valid_2d_nhwc(&input, &mut output.view_mut());
            } else {
                self.valid_2d(&input, &mut output.view_mut());
            }
        } else {
            let h_axis = self.patch.input_shape.h_axis();
            let non_valid_top = self.patch.pad_before[0].div_ceil(self.patch.kernel_strides[0]);
            let non_valid_bottom = self.patch.pad_after[0].div_ceil(self.patch.kernel_strides[0]);
            let non_valid_left = self.patch.pad_before[1].div_ceil(self.patch.kernel_strides[1]);
            let non_valid_right = self.patch.pad_after[1].div_ceil(self.patch.kernel_strides[1]);

            let start_non_valid_y = self.patch.output_spatial_shape[0] - non_valid_bottom;
            let start_non_valid_x = self.patch.output_spatial_shape[1] - non_valid_right;

            let mut valid_output = output.view_mut();
            valid_output
                .slice_axis_inplace(Axis(h_axis), (non_valid_top..start_non_valid_y).into());
            valid_output
                .slice_axis_inplace(Axis(h_axis + 1), (non_valid_left..start_non_valid_x).into());
            let mut valid_input = input.view();
            valid_input.slice_axis_inplace(
                Axis(h_axis),
                (non_valid_top * self.patch.kernel_strides[0]..).into(),
            );
            valid_input.slice_axis_inplace(
                Axis(h_axis + 1),
                (non_valid_left * self.patch.kernel_strides[1]..).into(),
            );
            if self.patch.input_shape.fmt == DataFormat::NHWC {
                self.valid_2d_nhwc(&valid_input, &mut valid_output);
            } else {
                self.valid_2d(&valid_input, &mut valid_output);
            }

            use ndarray::IntoDimension;
            let i = input.view().into_dimensionality()?;
            let visitor = self.patch.wrap(&i);
            output
                .slice_axis_mut(Axis(h_axis), (0..non_valid_top).into())
                .indexed_iter_mut()
                .for_each(|(coords, it)| {
                    *it = self.compute_one(&visitor, coords.into_dimension().slice());
                });
            output
                .slice_axis_mut(Axis(h_axis), (start_non_valid_y..).into())
                .indexed_iter_mut()
                .for_each(|(coords, it)| {
                    let mut coords = coords.into_dimension().into_dyn();
                    coords[h_axis] += start_non_valid_y;
                    *it = self.compute_one(&visitor, coords.slice());
                });
            output
                .slice_axis_mut(Axis(h_axis + 1), (0..non_valid_left).into())
                .indexed_iter_mut()
                .for_each(|(coords, it)| {
                    *it = self.compute_one(&visitor, coords.into_dimension().slice());
                });
            output
                .slice_axis_mut(Axis(h_axis + 1), (start_non_valid_x..).into())
                .indexed_iter_mut()
                .for_each(|(coords, it)| {
                    let mut coords = coords.into_dimension().into_dyn();
                    coords[h_axis + 1] += start_non_valid_x;
                    *it = self.compute_one(&visitor, coords.slice());
                });
        }
        Ok(output)
    }

    pub fn generic(&self, input: &ArrayViewD<T>) -> TractResult<ArrayD<T>> {
        let output_shape: TVec<usize> =
            self.patch.output_full_shape(self.patch.input_shape.c_dim());
        let input = input.view();
        let visitor = self.patch.wrap(&input);
        let output = ArrayD::from_shape_fn(&*output_shape, |coords| {
            self.compute_one(&visitor, coords.slice())
        });
        Ok(output)
    }

    fn compute_one<'v>(&self, visitor: &'v PatchVisitor<T>, coords: &[usize]) -> T {
        let pair = visitor
            .at(&coords)
            .map(|ov| ov.map(|v| (v, true)).unwrap_or((T::zero(), false)))
            .filter(|pair| pair.1 || self.count_include_pad)
            .fold((T::zero(), 0), |acc, pair| (acc.0 + pair.0, acc.1 + 1));
        pair.0 / (pair.1.as_())
    }
}

impl<T: Datum + Float> Op for FixedAvgPool<T>
where
    T: Datum + Float,
    usize: AsPrimitive<T>,
{
    fn name(&self) -> Cow<str> {
        format!("FixedAvgPool<{:?}>", T::datum_type()).into()
    }
}

impl<T> StatelessOp for FixedAvgPool<T>
where
    T: Datum + Float,
    usize: AsPrimitive<T>,
{
    fn eval(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let input = args_1!(inputs);
        let input = input.to_array_view::<T>()?;

        let result = if self.patch.kernel_spatial_shape.len() == 2 {
            self.two_d(&input.into_dimensionality()?)?.into_dyn()
        } else {
            self.generic(&input)?
        };

        Ok(tvec!(result.into()))
    }
}

impl<T> InferenceRulesOp for FixedAvgPool<T>
where
    T: Datum + Float,
    usize: AsPrimitive<T>,
{
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].datum_type, T::datum_type())?;
        s.equals(&outputs[0].datum_type, T::datum_type())?;
        s.equals(&outputs[0].rank, &inputs[0].rank)?;
        s.equals(&inputs[0].shape, ShapeFact::from(&*self.patch.input_shape.shape))?;
        let shape: TVec<usize> = self.patch.output_full_shape(self.patch.input_shape.c_dim());
        s.equals(&outputs[0].shape, ShapeFact::from(shape))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use proptest::*;
    use super::super::patches::test::patch_2d;

    pub fn patch_2d_and_data() -> BoxedStrategy<(Patch, Array4<f32>)> {
        patch_2d()
            .prop_flat_map(|p| {
                let len = p.input_shape.shape.iter().cloned().product();
                let vec = vec(-5.0..5.0f32, len..=len);
                (Just(p), vec)
            })
            .prop_map(|(p, v)| {
                let data = ndarray::ArrayD::from_shape_vec(&*p.input_shape.shape, v)
                    .unwrap()
                    .into_dimensionality()
                    .unwrap();
                (p, data)
            })
            .boxed()
    }

    proptest! {
        #[test]
        #[ignore]
        fn test_2d((p, d) in patch_2d_and_data()) {
            let op = FixedAvgPool::new(p, true);
            prop_assert_eq!(op.generic(&d.view().into_dyn()).unwrap(), op.two_d(&d.view()).unwrap().into_dyn())
        }
    }
}
