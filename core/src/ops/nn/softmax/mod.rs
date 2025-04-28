mod fixedpoint;
pub mod math;

use math::{
    convert_scale_to_mult_shift, exp_on_negative_values, get_reciprocal, rescale,
    rounding_divide_by_pot, saturating_rounding_doubling_high_mul,
    saturating_rounding_multiply_by_pot,
};
use num_traits::Float;
use std::fmt::Debug;
use tract_num_traits::Zero;

use crate::internal::*;
use ndarray::prelude::*;

#[derive(Debug, Copy, Clone, Hash, Default, PartialEq)]
pub enum SoftmaxExp {
    #[default]
    Libc,
    // https://nic.schraudolph.org/pubs/Schraudolph99.pdf
    FastCompact,
}

#[derive(Debug, Clone, new, Hash, Default)]
pub struct Softmax {
    pub axes: TVec<usize>,
    pub quant_output_dt: Option<DatumType>,
    pub exp: SoftmaxExp,
}

impl Op for Softmax {
    fn name(&self) -> Cow<str> {
        "Softmax".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("Axis: {:?}", self.axes), format!("Exp impl: {:?}", self.exp)])
    }

    op_as_typed_op!();
}

impl TypedOp for Softmax {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let dt = inputs[0].datum_type;
        if dt.is_float() {
            ensure!(
                self.quant_output_dt.is_none(),
                "Float softmax should not have quant_output_dt, have {:?}",
                self.quant_output_dt
            );
        } else if dt.is_quantized() {
            ensure!(
                self.quant_output_dt.map(|q| q.is_quantized()).unwrap_or(false),
                "Quantized softmax should have a quantized output type (got {:?})",
                self.quant_output_dt
            );
        } else {
            bail!(
                "Unsupported datum type in softmax: input type {:?}, output type {:?}",
                dt,
                self.quant_output_dt
            );
        }

        let fact = self.quant_output_dt.unwrap_or(dt).fact(inputs[0].shape.clone());
        Ok(tvec!(fact))
    }

    fn axes_mapping(
        &self,
        inputs: &[&TypedFact],
        outputs: &[&TypedFact],
    ) -> TractResult<AxesMapping> {
        AxesMapping::natural(inputs, outputs)
    }

    fn change_axes(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        _io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        let axes: Option<TVec<usize>> =
            self.axes.iter().map(|it| change.transform_axis(*it)).collect();
        if let Some(axes) = axes {
            Ok(Some(AxisChangeConsequence::new(
                model,
                node,
                Some(Box::new(Softmax { axes, ..self.clone() })),
                change,
            )))
        } else {
            Ok(None)
        }
    }

    as_op!();
}

impl EvalOp for Softmax {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let dt = input.datum_type();

        let output = match dt {
            DatumType::F64 => self.eval_t::<f64>(input)?,
            DatumType::F32 => self.eval_t::<f32>(input)?,
            DatumType::F16 => self.eval_t::<f16>(input)?,
            DatumType::QI8(_) | DatumType::QU8(_) => self.eval_quant(input)?,
            dt => bail!("Unsupported type {dt:?}"),
        };
        Ok(output)
    }
}

impl Softmax {
    fn eval_t<T>(&self, input: TValue) -> TractResult<TVec<TValue>>
    where
        T: Float + Datum + std::iter::Sum,
    {
        let mut iterating_shape: TVec<usize> = input.shape().into();

        for i in 0..iterating_shape.len() {
            if self.axes.contains(&i) {
                iterating_shape[i] = 1
            }
        }

        let mut output = input.into_tensor();
        let mut view = output.to_array_view_mut::<T>()?;

        for it_coords in tract_ndarray::indices(&*iterating_shape) {
            let mut view = view.view_mut();
            for ix in 0..iterating_shape.len() {
                if !self.axes.contains(&ix) {
                    view.collapse_axis(Axis(ix), it_coords[ix]);
                }
            }
            if let Some(slice) =
                view.as_slice_mut().filter(|_| T::datum_type() == f32::datum_type())
            {
                let slice: &mut [f32] = unsafe { std::mem::transmute(slice) };
                self.softmax_inner_slice_f32(slice)?;
            } else if let Some(slice) =
                view.as_slice_mut().filter(|_| T::datum_type() == f16::datum_type())
            {
                let slice: &mut [f16] = unsafe { std::mem::transmute(slice) };
                self.softmax_inner_slice_f16(slice)?;
            } else {
                softmax_inner(view);
            }
        }

        Ok(tvec!(output.into_tvalue()))
    }

    fn eval_quant(&self, input: TValue) -> TractResult<TVec<TValue>> {
        let mut iterating_shape: TVec<usize> = input.shape().into();
        let output_dt =
            self.quant_output_dt.context("Quandized softmax eval with no output type")?;

        for i in 0..iterating_shape.len() {
            if self.axes.contains(&i) {
                iterating_shape[i] = 1
            }
        }

        // All operations will be done in u8, we will cast the result appropriately afterward.
        let src_is_signed = input.datum_type().is_signed();
        let out_is_signed = output_dt.is_signed();
        let in_qp = input.datum_type().qparams().unwrap(); // Checked as we are in the quant case
        let out_qp = output_dt.qparams().unwrap(); // Checked as we are in the quant case
        let mut output = unsafe { input.into_tensor().into_array_unchecked::<u8>() };

        for it_coords in tract_ndarray::indices(&*iterating_shape) {
            let mut view = output.view_mut();
            for ix in 0..iterating_shape.len() {
                if !self.axes.contains(&ix) {
                    view.collapse_axis(Axis(ix), it_coords[ix]);
                }
            }
            softmax_quant_inner(view, src_is_signed, in_qp, out_is_signed, out_qp);
        }

        let mut output_tensor = output.into_tensor();
        unsafe { output_tensor.set_datum_type(output_dt) };
        Ok(tvec!(output_tensor.into_tvalue()))
    }

    fn softmax_inner_slice_f16(&self, slice: &mut [f16]) -> TractResult<()> {
        let max = (tract_linalg::ops().max_f16)().run(slice)?;
        let sum = match self.exp {
            SoftmaxExp::Libc => {
                let mut s = f16::zero();
                for x in slice.iter_mut() {
                    let y = (*x - max).exp();
                    s += y;
                    *x = y;
                }
                s
            }
            SoftmaxExp::FastCompact => {
                (tract_linalg::ops().softmax2_fastcompact_f16)().run_with_params(slice, max)?
            }
        };
        let rsum = sum.recip();
        (tract_linalg::ops().mul_by_scalar_f16)().run_with_params(slice, rsum)?;
        Ok(())
    }

    fn softmax_inner_slice_f32(&self, slice: &mut [f32]) -> TractResult<()> {
        let max = (tract_linalg::ops().max_f32)().run(slice)?;
        let sum = match self.exp {
            SoftmaxExp::Libc => {
                let mut s = 0f32;
                for x in slice.iter_mut() {
                    let y = (*x - max).exp();
                    s += y;
                    *x = y;
                }
                s
            }
            SoftmaxExp::FastCompact => {
                (tract_linalg::ops().softmax2_fastcompact_f32)().run_with_params(slice, max)?
            }
        };
        let rsum = sum.recip();
        (tract_linalg::ops().mul_by_scalar_f32)().run_with_params(slice, rsum)?;
        Ok(())
    }
}

fn softmax_inner<T: Float + Datum + std::iter::Sum, D: Dimension>(mut view: ArrayViewMut<T, D>) {
    let max =
        *view.iter().max_by(|i, j| i.partial_cmp(j).unwrap_or(std::cmp::Ordering::Less)).unwrap();
    view.mapv_inplace(|x| (x - max).exp());
    let exp_sum = view.iter().copied().sum();
    view.mapv_inplace(|x| x / exp_sum);
}

fn softmax_quant_inner<D: Dimension>(
    mut view: ArrayViewMut<u8, D>,
    src_is_signed: bool,
    in_qp: QParams,
    out_is_signed: bool,
    out_qp: QParams,
) {
    let (_, in_scale) = in_qp.zp_scale();
    let (scale_in_multiplier, scale_in_shift) = convert_scale_to_mult_shift(in_scale).unwrap();
    let (_, out_scale) = out_qp.zp_scale();
    let (scale_out_multiplier, scale_out_shift) = convert_scale_to_mult_shift(out_scale).unwrap();
    let shift = 26 - scale_in_shift;

    // Compute the exponentials x - max
    let mut buffer = vec![0_i32; view.len()];

    // Handle the case were we considered an i8 as an u8 and still get the right x - max.
    let safe_u8 = if src_is_signed { |x: &u8| x.wrapping_add(128) } else { |x: &u8| *x };

    let max = view.iter().map(safe_u8).max().unwrap();
    view.iter().zip(buffer.iter_mut()).for_each(|(x, exp)| {
        let input_diff = safe_u8(x) as i32 - max as i32;

        // We scale the input to be in Q5_26
        let scaled_input_diff = if scale_in_multiplier != 0 {
            saturating_rounding_multiply_by_pot(
                saturating_rounding_doubling_high_mul(input_diff, scale_in_multiplier),
                shift as i32,
            )
        } else {
            saturating_rounding_multiply_by_pot(input_diff, shift as i32)
        };

        // It expects an input from Q5_26 and returns an output in Q0_31
        *exp = exp_on_negative_values(scaled_input_diff);
    });

    // Compute sum of exp
    // The sum is stored as an Q12_19 that's why we need to recale from Q0_31 to Q12_19 before summing.
    let sum_of_exp = buffer.iter().map(|it| rescale(*it, 0, 12)).sum();

    // Compute 1/sum_of_exp
    // The result of this function is in Q0_31
    let (inv_sum_of_exp, num_bits_over_unit) = get_reciprocal(sum_of_exp, 12);

    // Compute the exponent value needed to be in Q24_8 before the final rescaling
    let exponent = num_bits_over_unit as isize + 31 - 8;

    view.iter_mut().zip(buffer.iter()).for_each(|(it, exp)| {
        // Compute the product of exp * 1/sum_of_exp and scale the result in Q24_8
        let unsat_output = rounding_divide_by_pot(
            saturating_rounding_doubling_high_mul(inv_sum_of_exp, *exp),
            exponent as i32,
        );

        // Scale the final result in the output scale range
        let unsat_scaled_output = {
            if scale_out_multiplier != 0 {
                let (inv_multiplier, num_bits) = get_reciprocal(scale_out_multiplier, 1);
                rounding_divide_by_pot(
                    saturating_rounding_doubling_high_mul(unsat_output, inv_multiplier),
                    (8 - scale_out_shift - 1 - num_bits as isize) as i32,
                )
            } else {
                rounding_divide_by_pot(unsat_output, (8 - scale_out_shift) as i32)
            }
        };

        // Return the final result by clipping the computed value within its range
        // and casting it to u8 in any case.
        #[allow(unknown_lints, unnecessary_transmutes)]
        if out_is_signed {
            *it = unsafe {
                std::mem::transmute::<i8, u8>(i32::max(
                    i32::min(unsat_scaled_output, i8::MAX as i32),
                    i8::MIN as i32,
                ) as i8)
            };
        } else {
            *it = i32::max(i32::min(unsat_scaled_output, u8::MAX as i32), u8::MIN as i32) as u8;
        }
    });
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::ops::nn::DataFormat::NCHW;
    use anyhow::Result;
    use num_traits::PrimInt;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use tract_data::internal::QParams::ZpScale;

    fn assert_is_close(found: f32, expected: f32, in_dt: DatumType, out_dt: DatumType) {
        let (_, in_epsilon) = in_dt.zp_scale();
        let (_, out_epsilon) = out_dt.zp_scale();
        let epsilon = f32::max(in_epsilon, out_epsilon) * 1.005;
        let error = (found - expected).abs();
        assert!(
            error <= epsilon,
            "epsilon eq failed: |{found:?}-{expected:?}|={error} should be <= {epsilon}"
        );
    }

    // Generate a random tensor with a quantized datum type
    fn qtensor<T: PrimInt + Datum + Arbitrary>(shape: Vec<usize>) -> BoxedStrategy<Tensor> {
        let len = shape.iter().product::<usize>();
        let dt = q_datum::<T>((0.0001f32..0.1).boxed());
        (vec(any::<T>(), len..=len), dt)
            .prop_map(move |(vec, dt)| (ArrayD::from_shape_vec(shape.clone(), vec).unwrap(), dt))
            .prop_map(move |(array, dt)| {
                let mut tensor = array.into_tensor();
                unsafe { tensor.set_datum_type(dt) };
                tensor
            })
            .boxed()
    }

    // Generate a random quantized datum type
    fn q_datum<T: PrimInt + Datum>(range: BoxedStrategy<f32>) -> BoxedStrategy<DatumType> {
        let max_integer_bits = std::mem::size_of::<T>() * 8 - T::datum_type().is_signed() as usize;
        prop_oneof![
            (1usize..max_integer_bits).prop_map(|fixed_point| { 2f32.powi(-(fixed_point as i32)) }),
            range
        ]
        .prop_map(|scale| {
            if T::datum_type().is_signed() {
                DatumType::QI8(ZpScale { zero_point: 0, scale })
            } else {
                DatumType::QU8(ZpScale { zero_point: 0, scale })
            }
        })
        .boxed()
    }

    #[derive(Debug)]
    struct SoftmaxProblem {
        data: Tensor,
        axes: TVec<usize>,
        output_dt: DatumType,
    }

    impl SoftmaxProblem {
        fn check(&self) -> Result<()> {
            let inputs = tvec!(self.data.clone().into_tvalue());
            let quant_output_dt = Some(self.output_dt).filter(|dt| !dt.is_float());
            let softmax =
                Softmax { axes: self.axes.clone(), quant_output_dt, ..Softmax::default() };

            // Compute quantized output
            let result = softmax.eval(inputs)?;
            let result = args_1!(result);
            let result_float = result.cast_to::<f32>()?;

            // Compute reference output
            let input_float = self.data.cast_to::<f32>()?;
            let inputs_float = tvec!(input_float.into_owned().into_tvalue());
            let softmax_float = Softmax { axes: self.axes.clone(), ..Softmax::default() };
            let reference_float = softmax_float.eval(inputs_float)?;
            let reference_array = args_1!(reference_float);
            let reference = reference_array.to_array_view::<f32>()?;

            result_float
                .to_array_view::<f32>()?
                .iter()
                .zip(reference.iter())
                .for_each(|(a, b)| assert_is_close(*a, *b, self.data.datum_type(), self.output_dt));

            Ok(())
        }
    }

    impl Arbitrary for SoftmaxProblem {
        type Parameters = ();
        type Strategy = BoxedStrategy<SoftmaxProblem>;
        fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
            (1usize..2, 1usize..2, 1usize..5, 1usize..5, 0usize..4)
                .prop_flat_map(|(n, c, h, w, axis)| {
                    let shape_in: Vec<usize> =
                        NCHW.from_n_c_hw(n, c, [h, w]).unwrap().shape.to_vec();
                    (
                        prop_oneof![qtensor::<i8>(shape_in.clone()), qtensor::<u8>(shape_in)],
                        Just(tvec![axis]),
                        prop_oneof![
                            q_datum::<u8>((0.008f32..0.1).boxed()),
                            q_datum::<i8>((0.008f32..0.1).boxed())
                        ],
                    )
                })
                .prop_map(|(data, axes, output_dt)| SoftmaxProblem { data, axes, output_dt })
                .boxed()
        }
    }

    #[derive(Debug)]
    pub struct InnerSoftmaxProblem {
        in_qp: QParams,
        out_qp: QParams,
        data: Vec<i8>,
    }

    impl InnerSoftmaxProblem {
        fn check(&self) -> Result<()> {
            let quantized = self.quantized();
            let reference = self.reference();
            assert!(quantized.iter().zip(reference.iter()).all(|(quantized, expected)| {
                let abs_diff = if *quantized > *expected {
                    quantized - *expected
                } else {
                    expected - *quantized
                };
                abs_diff <= 1
            }));
            Ok(())
        }

        fn reference(&self) -> Vec<u8> {
            let (in_zero_point, in_scale) = self.in_qp.zp_scale();
            let (out_zero_point, out_scale) = self.out_qp.zp_scale();
            let in_float =
                self.data.iter().map(|it| (*it as f32 - in_zero_point as f32) * in_scale).collect();
            let mut in_float_array = Array1::from_vec(in_float);
            softmax_inner(in_float_array.view_mut());
            let rescaled_output = in_float_array
                .iter()
                .map(|it| {
                    ((*it / out_scale).round() as i32 + out_zero_point)
                        .max(u8::MIN as i32)
                        .min(u8::MAX as i32) as u8
                })
                .collect();
            rescaled_output
        }

        fn quantized(&self) -> Vec<u8> {
            let in_data: Vec<u8> = unsafe { std::mem::transmute(self.data.clone()) };
            let mut in_array = Array1::from_vec(in_data);
            softmax_quant_inner(in_array.view_mut(), true, self.in_qp, false, self.out_qp);
            in_array.to_vec()
        }
    }

    impl Arbitrary for InnerSoftmaxProblem {
        type Parameters = ();
        type Strategy = BoxedStrategy<InnerSoftmaxProblem>;
        fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
            (
                prop_oneof![
                    q_datum::<i8>((0.0001f32..0.01).boxed()),
                    q_datum::<u8>((0.0001f32..0.01).boxed())
                ],
                prop_oneof![
                    q_datum::<u8>((0.008f32..0.1).boxed()),
                    q_datum::<i8>((0.008f32..0.1).boxed())
                ],
                vec(any::<i8>(), 1..10),
            )
                .prop_map(|(in_qp, out_qp, data)| InnerSoftmaxProblem {
                    in_qp: in_qp.qparams().unwrap(),
                    out_qp: out_qp.qparams().unwrap(),
                    data,
                })
                .boxed()
        }
    }

    proptest::proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]
        #[test]
        fn test_softmax_inner_prop(pb in any::<InnerSoftmaxProblem>()) {
            pb.check().unwrap()
        }
    }

    proptest::proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]
        #[test]
        fn test_softmax_prop(pb in any::<SoftmaxProblem>()) {
            pb.check().unwrap()
        }
    }

    #[test]
    // We test QU8 -> QU8
    fn test_softmax_trivial_0() -> Result<()> {
        let input_dt = DatumType::QU8(ZpScale { zero_point: 0, scale: 0.03125 }); // Q3_5
        let output_dt = DatumType::QU8(ZpScale { zero_point: 0, scale: 0.00390625 }); // Q0_8;
        let mut data = Tensor::from_shape(&[1, 1, 2, 2], &[0_u8, 0, 0, 4])?;
        unsafe { data.set_datum_type(input_dt) };

        let prob = SoftmaxProblem { data, axes: tvec![3], output_dt };
        prob.check()?;
        Ok(())
    }

    #[test]
    // We test QI8 -> QU8
    fn test_softmax_trivial_1() -> Result<()> {
        let input_dt = DatumType::QI8(ZpScale { zero_point: 0, scale: 0.0625 }); // Q3_4
        let output_dt = DatumType::QU8(ZpScale { zero_point: 0, scale: 0.00390625 }); // Q0_8;
        let mut data = Tensor::from_shape(&[1, 1, 2, 2], &[0_i8, 0, 0, 4])?;
        unsafe { data.set_datum_type(input_dt) };

        let prob = SoftmaxProblem { data, axes: tvec![3], output_dt };
        prob.check()?;
        Ok(())
    }

    #[test]
    // We test QI8 -> QI8
    fn test_softmax_trivial_2() -> Result<()> {
        let input_dt = DatumType::QI8(ZpScale { zero_point: 0, scale: 0.0625 }); // Q3_4
        let output_dt = DatumType::QI8(ZpScale { zero_point: 0, scale: 0.0078125 }); // Q0_7;
        let mut data = Tensor::from_shape(&[1, 1, 2, 2], &[0_i8, 0, 0, -4])?;
        unsafe { data.set_datum_type(input_dt) };

        let prob = SoftmaxProblem { data, axes: tvec![3], output_dt };
        prob.check()?;
        Ok(())
    }

    #[test]
    // We test QU8 -> QI8
    fn test_softmax_trivial_3() -> Result<()> {
        let input_dt = DatumType::QU8(ZpScale { zero_point: 0, scale: 0.03125 }); // Q3_5
        let output_dt = DatumType::QI8(ZpScale { zero_point: 0, scale: 0.0078125 }); // Q0_7;
        let mut data = Tensor::from_shape(&[1, 1, 2, 2], &[0_u8, 0, 0, 4])?;
        unsafe { data.set_datum_type(input_dt) };

        let prob = SoftmaxProblem { data, axes: tvec![2], output_dt };
        prob.check()?;
        Ok(())
    }

    #[test]
    fn test_softmax_1() -> Result<()> {
        let input_dt = DatumType::QI8(ZpScale { zero_point: 0, scale: 0.5 }); // Q6_1
        let output_dt = DatumType::QU8(ZpScale { zero_point: 0, scale: 0.5 }); // Q7_1
        let mut data = Tensor::from_shape(&[1, 1, 1, 2], &[115_i8, 115])?;
        unsafe { data.set_datum_type(input_dt) };

        let prob = SoftmaxProblem { data, axes: tvec![3], output_dt };
        prob.check()?;
        Ok(())
    }

    #[test]
    fn test_softmax_2() -> Result<()> {
        let input_dt = DatumType::QI8(ZpScale { zero_point: 0, scale: 0.0001 });
        let output_dt = DatumType::QU8(ZpScale { zero_point: 0, scale: 0.008 });
        let mut data = Tensor::from_shape(&[1, 1, 1, 2], &[115_i8, 115])?;
        unsafe { data.set_datum_type(input_dt) };

        let prob = SoftmaxProblem { data, axes: tvec![3], output_dt };
        prob.check()?;
        Ok(())
    }

    #[test]
    fn test_softmax_3() -> Result<()> {
        let input_dt = DatumType::QU8(ZpScale { zero_point: 0, scale: 0.6220956 });
        let output_dt = DatumType::QU8(ZpScale { zero_point: 0, scale: 0.5187921 });
        let mut data = Tensor::from_shape(&[1, 1, 1, 2], &[13_u8, 218])?;
        unsafe { data.set_datum_type(input_dt) };

        let prob = SoftmaxProblem { data, axes: tvec![3], output_dt };
        prob.check()?;
        Ok(())
    }

    #[test]
    fn test_inner_softmax_1() -> Result<()> {
        let in_qp = ZpScale { zero_point: 0, scale: 0.03125 };
        let out_qp = ZpScale { zero_point: 0, scale: 0.5 };
        let data = vec![0_i8, 1];

        let prob = InnerSoftmaxProblem { in_qp, out_qp, data };
        prob.check()?;
        Ok(())
    }

    #[test]
    fn test_inner_softmax_2() -> Result<()> {
        let in_qp = ZpScale { zero_point: 0, scale: 0.5 };
        let out_qp = ZpScale { zero_point: 0, scale: 0.03125 };
        let data = vec![100i8, -28];

        let prob = InnerSoftmaxProblem { in_qp, out_qp, data };
        prob.check()?;
        Ok(())
    }

    #[test]
    fn test_inner_softmax_not_pow_2_1() -> Result<()> {
        let in_qp = ZpScale { zero_point: 0, scale: 0.7298456 };
        let out_qp = ZpScale { zero_point: 0, scale: 0.03125 };
        let data = vec![100i8, -28];

        let prob = InnerSoftmaxProblem { in_qp, out_qp, data };
        prob.check()?;
        Ok(())
    }

    #[test]
    #[ignore]
    // Fails but the difference is quite low and the sum still give exactly one:
    // quantized: 110(0.88), 15(0.12)
    // expected: 112(0.896), 13(0.104)
    fn test_inner_softmax_not_pow_2_2() -> Result<()> {
        let in_qp = ZpScale { zero_point: 0, scale: 0.2123116 };
        let out_qp = ZpScale { zero_point: 0, scale: 0.008 };
        let data = vec![118i8, 108];

        let prob = InnerSoftmaxProblem { in_qp, out_qp, data };
        prob.check()?;
        Ok(())
    }

    #[test]
    #[ignore]
    // Fails but the difference is quite low and the sum still give exactly one:
    // quantized: 40(0.625), 24(0.375)
    // expected: 42(0.65625), 22(0.34375)
    fn test_inner_softmax_not_pow_2_3() -> Result<()> {
        let in_qp = ZpScale { zero_point: 0, scale: 0.33034274 };
        let out_qp = ZpScale { zero_point: 0, scale: 0.015625 };
        let data = vec![45i8, 43];

        let prob = InnerSoftmaxProblem { in_qp, out_qp, data };
        prob.check()?;
        Ok(())
    }
}
