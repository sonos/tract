use crate::internal::*;
use ndarray::prelude::*;
use num_traits::{AsPrimitive, Float};

use crate::ops::cnn::pools::PoolSpec;
use crate::ops::cnn::Patch;
use crate::ops::nn::{DataFormat, DataShape};

#[derive(Debug, Clone, new, Default)]
pub struct AvgPool {
    pool_spec: PoolSpec,
    count_include_pad: bool,
}

impl AvgPool {
    fn eval_t<T>(&self, inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>>
    where
        T: Datum + Float,
        usize: AsPrimitive<T>,
    {
        let (input_shape, patch, output_shape) = self.pool_spec.compute_geo(inputs[0].shape());
        FixedAvgPool::new(patch, input_shape, output_shape, self.count_include_pad).eval(inputs)
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
            let (input_shape, patch, output_shape) = self.pool_spec.compute_geo(shape);
            let dt = inputs[0].datum_type;
            fn fixed<T>(
                patch: Patch,
                input_shape: DataShape,
                output_shape: DataShape,
                count_include_pad: bool,
            ) -> Box<Op>
            where
                T: Datum + Float,
                usize: AsPrimitive<T>,
            {
                Box::new(FixedAvgPool::new(patch, input_shape, output_shape, count_include_pad))
            }
            let op = dispatch_floatlike!(fixed(dt)(
                patch,
                input_shape,
                output_shape,
                self.count_include_pad
            ));
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
        self.pool_spec.rules_for_shape(s, inputs, outputs)
    }
}

#[derive(Debug, Clone, new)]
pub struct FixedAvgPool<T>
where
    T: Datum + Float,
    usize: AsPrimitive<T>,
{
    patch: Patch,
    input_shape: DataShape,
    output_shape: DataShape,
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
            let out_h = output.shape()[self.input_shape.hw_axes()][0];
            let out_w = output.shape()[self.input_shape.hw_axes()][1];
            let stride_in_y = input.strides()[self.input_shape.hw_axes()][0]
                * self.patch.spec.strides[0] as isize;
            let stride_in_x = input.strides()[self.input_shape.hw_axes()][1]
                * self.patch.spec.strides[1] as isize;
            let stride_out_y = output.strides()[self.input_shape.hw_axes()][0];
            let stride_out_x = output.strides()[self.input_shape.hw_axes()][1];
            let k_len = self.patch.spec.kernel_shape.iter().cloned().product::<usize>().as_();
            for i in 0..self.input_shape.n() {
                let p_in =
                    input.slice_axis(Axis(self.input_shape.n_axis()), (i..=i).into()).as_ptr();
                let p_out = output
                    .slice_axis_mut(Axis(self.input_shape.n_axis()), (i..=i).into())
                    .as_mut_ptr();
                for y in 0..out_h {
                    let p_in = p_in.offset(stride_in_y * y as isize);
                    let p_out = p_out.offset(stride_out_y * y as isize);
                    for x in 0..out_w {
                        let p_in = p_in.offset(stride_in_x * x as isize);
                        let p_out = p_out.offset(stride_out_x * x as isize);
                        let max_c = self.input_shape.c();
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
            let out_h = self.output_shape.hw_dims()[0];
            let out_w = self.output_shape.hw_dims()[1];
            let stride_in_c = self.input_shape.c_stride() as isize;
            let stride_in_y = input.strides()[self.input_shape.hw_axes()][0]
                * self.patch.spec.strides[0] as isize;
            let stride_in_x = input.strides()[self.input_shape.hw_axes()][1]
                * self.patch.spec.strides[1] as isize;
            let stride_out_c = output.strides()[self.input_shape.c_axis()] as isize;
            let stride_out_y = output.strides()[self.input_shape.hw_axes()][0];
            let stride_out_x = output.strides()[self.input_shape.hw_axes()][1];
            let k_len = self.patch.spec.kernel_shape.iter().cloned().product::<usize>().as_();
            for i in 0..self.input_shape.n() {
                let p_in_i =
                    input.slice_axis(Axis(self.input_shape.n_axis()), (i..=i).into()).as_ptr();
                let p_out_i = output
                    .slice_axis_mut(Axis(self.output_shape.n_axis()), (i..=i).into())
                    .as_mut_ptr();
                for c in 0..self.input_shape.c() {
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
        let mut output =
            unsafe { ArrayD::uninitialized(&*self.output_shape.shape).into_dimensionality()? };
        if self.input_shape.fmt == DataFormat::NHWC {
            self.valid_2d_nhwc(&input, &mut output.view_mut());
        } else {
            self.valid_2d(&input, &mut output.view_mut());
        }
        Ok(output)
        /*
        let h_axis = self.input_shape.h_axis();
        let non_valid_top = self.patch.pad_before[0].div_ceil(self.patch.spec.strides[0]);
        let non_valid_bottom = self.patch.pad_after[0].div_ceil(self.patch.spec.strides[0]);
        let non_valid_left = self.patch.pad_before[1].div_ceil(self.patch.spec.strides[1]);
        let non_valid_right = self.patch.pad_after[1].div_ceil(self.patch.spec.strides[1]);

        let start_non_valid_y = self.patch.output_shape[0] - non_valid_bottom;
        let start_non_valid_x = self.patch.output_shape[1] - non_valid_right;

        let mut valid_output = output.view_mut();
        valid_output
            .slice_axis_inplace(Axis(h_axis), (non_valid_top..start_non_valid_y).into());
        valid_output
            .slice_axis_inplace(Axis(h_axis + 1), (non_valid_left..start_non_valid_x).into());
        let mut valid_input = input.view();
        valid_input.slice_axis_inplace(
            Axis(h_axis),
            (non_valid_top * self.patch.spec.strides[0]..).into(),
        );
        valid_input.slice_axis_inplace(
            Axis(h_axis + 1),
            (non_valid_left * self.patch.spec.strides[1]..).into(),
        );
        if self.input_shape.fmt == DataFormat::NHWC {
            self.valid_2d_nhwc(&valid_input, &mut valid_output);
        } else {
            self.valid_2d(&valid_input, &mut valid_output);
        }

        use ndarray::IntoDimension;
        let i = input.view().into_dimensionality()?;
        let visitor = self.patch.wrap(&i);
        let ptr = i.as_ptr();
        output
            .slice_axis_mut(Axis(h_axis), (0..non_valid_top).into())
            .indexed_iter_mut()
            .for_each(|(coords, it)| {
                *it = self.compute_one(ptr, &visitor, coords.into_dimension().slice());
            });
        output
            .slice_axis_mut(Axis(h_axis), (start_non_valid_y..).into())
            .indexed_iter_mut()
            .for_each(|(coords, it)| {
                let mut coords = coords.into_dimension().into_dyn();
                coords[h_axis] += start_non_valid_y;
                *it = self.compute_one(ptr, &visitor, coords.slice());
            });
        output
            .slice_axis_mut(Axis(h_axis + 1), (0..non_valid_left).into())
            .indexed_iter_mut()
            .for_each(|(coords, it)| {
                *it = self.compute_one(ptr, &visitor, coords.into_dimension().slice());
            });
        output
            .slice_axis_mut(Axis(h_axis + 1), (start_non_valid_x..).into())
            .indexed_iter_mut()
            .for_each(|(coords, it)| {
                let mut coords = coords.into_dimension().into_dyn();
                coords[h_axis + 1] += start_non_valid_x;
                *it = self.compute_one(ptr, &visitor, coords.slice());
            });
            */
    }

    pub fn generic(&self, input: &ArrayViewD<T>) -> TractResult<ArrayD<T>> {
        let input = input.view();
        let ptr = input.as_ptr();
        let output = ArrayD::from_shape_fn(&*self.output_shape.shape, |coords| {
            self.compute_one(ptr, coords.slice())
        });
        Ok(output)
    }

    fn compute_one<'v>(&self, input: *const T, coords: &[usize]) -> T {
        unsafe {
            assert_eq!(coords.len(), self.patch.spec.kernel_shape.len() + 2);
            let shape = &self.input_shape;
            let input = input.offset((shape.n_stride() * coords[shape.n_axis()]) as isize);
            let input = input.offset((shape.c_stride() * coords[shape.c_axis()]) as isize);
            let pair = self
                .patch
                .at(&coords[shape.hw_axes()])
                .map(|offset| {
                    offset.map(|offset| (*input.offset(offset), true)).unwrap_or((T::zero(), false))
                })
                .filter(|pair| pair.1 || self.count_include_pad)
                .fold((T::zero(), 0), |acc, pair| (acc.0 + pair.0, acc.1 + 1));
            pair.0 / (pair.1.as_())
        }
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

        let result =
        /*
        if self.patch.spec.kernel_shape.len() == 2 && !self.patch.padded {
            self.two_d(&input.into_dimensionality()?)?.into_dyn()
        } else {
        */
            self.generic(&input)?
        //}
        ;

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
        s.equals(&inputs[0].shape, ShapeFact::from(&*self.input_shape.shape))?;
        s.equals(&outputs[0].shape, ShapeFact::from(&*self.output_shape.shape))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::cnn::patches::test::patch_2d;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use proptest::*;

    pub fn patch_2d_and_data() -> BoxedStrategy<(DataShape, Patch, Array4<f32>)>
    {
        patch_2d()
            .prop_flat_map(|(i, p)| {
                let len = i.shape.iter().cloned().product();
                let vec = vec(-5.0..5.0f32, len..=len);
                (Just(i), Just(p), vec)
            })
            .prop_map(|(i, p, v)| {
                let data = ndarray::ArrayD::from_shape_vec(&*i.shape, v)
                    .unwrap()
                    .into_dimensionality()
                    .unwrap();
                (i, p, data)
            })
            .boxed()
    }

    proptest! {
        #[test]
        #[ignore]
        fn test_2d((i, p, d) in patch_2d_and_data()) {
            let o = i.fmt.from_n_c_hw(i.n(), i.c(), &*p.output_shape);
            let op = FixedAvgPool::new(p, i, o, true);
            prop_assert_eq!(op.generic(&d.view().into_dyn()).unwrap(), op.two_d(&d.view()).unwrap().into_dyn())
        }
    }
}
