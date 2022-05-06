mod fixedpoint;
pub mod math;

use math::{
    exp_on_negative_values, get_reciprocal, rescale, rounding_divide_by_pot,
    saturating_rounding_doubling_high_mul,
};
use num_traits::Float;
use std::fmt::Debug;

use crate::internal::*;
use ndarray::prelude::*;

#[derive(Debug, Clone, new, Hash)]
pub struct Softmax {
    pub axes: TVec<usize>,
    pub output_dt: DatumType,
}

impl_dyn_hash!(Softmax);

impl Op for Softmax {
    fn name(&self) -> Cow<str> {
        "Softmax".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("Axis: {:?}", self.axes)])
    }

    op_core_mir!();
    op_as_typed_op!();
}

impl TypedOp for Softmax {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let dt = inputs[0].datum_type;
        if dt.is_float() {
            ensure!(
                dt == self.output_dt,
                "Softmax input {:?} and output {:?} types in float case should be equal",
                dt,
                self.output_dt
            );
        } else if dt.is_quantized() {
            ensure!(
                self.output_dt.is_quantized(),
                "Quantized softmax must have input {:?} and output {:?} quantized ",
                dt,
                self.output_dt
            );
        } else {
            bail!(
                "Unsupported datum type in softmax: input type {:?}, output type {:?}",
                dt,
                self.output_dt
            );
        }

        let fact = inputs[0].clone();
        Ok(tvec!(fact))
    }

    fn invariants(
        &self,
        inputs: &[&TypedFact],
        _outputs: &[&TypedFact],
    ) -> TractResult<Invariants> {
        let axes = (0..inputs[0].rank()).map(|axis| AxisInfo::simple(axis)).collect();
        Ok(axes)
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
                Some(Box::new(Softmax { axes, output_dt: self.output_dt })),
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

    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        let dt = input.datum_type();

        let output = match dt {
            DatumType::F64 => self.eval_t::<f64>(input)?,
            DatumType::F32 => self.eval_t::<f32>(input)?,
            DatumType::F16 => self.eval_t::<f16>(input)?,
            DatumType::QI8(_) | DatumType::QU8(_) => self.eval_quant_t(input)?,
            dt => bail!("Unsupported type {:?}", dt),
        };
        Ok(output)
    }
}

impl Softmax {
    fn eval_t<T>(&self, input: Arc<Tensor>) -> TractResult<TVec<Arc<Tensor>>>
    where
        T: Float + Datum + std::iter::Sum,
    {
        let mut iterating_shape: TVec<usize> = input.shape().into();

        for i in 0..iterating_shape.len() {
            if self.axes.contains(&i) {
                iterating_shape[i] = 1
            }
        }

        let mut output = input.into_tensor().into_array::<T>()?;

        for it_coords in tract_ndarray::indices(&*iterating_shape) {
            let mut view = output.view_mut();
            for ix in 0..iterating_shape.len() {
                if !self.axes.contains(&ix) {
                    view.collapse_axis(Axis(ix), it_coords[ix]);
                }
            }

            let max = *view.iter().max_by(|i, j| i.partial_cmp(j).unwrap()).unwrap();
            view.mapv_inplace(|x| (x - max).exp());
            let exp_sum = view.iter().map(|it| *it).sum();
            view.mapv_inplace(|x| x / exp_sum);
        }

        Ok(tvec!(output.into_arc_tensor()))
    }

    fn eval_quant_t(&self, input: Arc<Tensor>) -> TractResult<TVec<Arc<Tensor>>> {
        let mut iterating_shape: TVec<usize> = input.shape().into();

        for i in 0..iterating_shape.len() {
            if self.axes.contains(&i) {
                iterating_shape[i] = 1
            }
        }

        // All operations will be done in u8, we will cast the result appropriately afterward.
        let src_is_signed = input.as_ref().datum_type().is_signed();
        let src_fixed_point = fixed_point(input.datum_type())?;
        let dst_is_signed = self.output_dt.is_signed();
        let dst_fixed_point = fixed_point(self.output_dt)?;
        let mut output = unsafe { input.into_tensor().into_array_unchecked::<u8>() };

        for it_coords in tract_ndarray::indices(&*iterating_shape) {
            let mut view = output.view_mut();
            for ix in 0..iterating_shape.len() {
                if !self.axes.contains(&ix) {
                    view.collapse_axis(Axis(ix), it_coords[ix]);
                }
            }

            softmax_quant_inner(
                view,
                src_is_signed,
                src_fixed_point,
                dst_is_signed,
                dst_fixed_point,
            );
        }

        let mut output_tensor = output.into_tensor();
        unsafe { output_tensor.set_datum_type(self.output_dt) };
        Ok(tvec!(Arc::new(output_tensor)))
    }
}

fn fixed_point(dt: DatumType) -> TractResult<usize> {
    let max_fixed_point = dt.size_of() * 8 - dt.is_signed() as usize;
    match dt {
        DatumType::QI8(_) | DatumType::QU8(_) => {
            let (_, scale) = dt.zp_scale();
            if (scale.log2() - scale.log2().round()).abs() <= 3. * f32::EPSILON {
                let fixed_point = -scale.log2().round() as usize;

                if fixed_point > max_fixed_point {
                    bail!("Quantization scale require too much precision")
                } else {
                    Ok(fixed_point)
                }
            } else {
                bail!("Softmax only support quantization parameter with zp=0 & scale = 1/2^n")
            }
        }
        _ => bail!("Fixed point can only be extracted from quantized datum types"),
    }
}

// TODO: support arbitraty scale with QScale parameters
// fn softmax_quant_inner<D: Dimension>(mut view: ArrayViewMut<i8, D>, is_signed: bool,  in_scale: QScale, out_scale: QScale)
fn softmax_quant_inner<D: Dimension>(
    mut view: ArrayViewMut<u8, D>,
    src_is_signed: bool,
    src_fixed_point: usize,
    dst_is_signed: bool,
    dst_fixed_point: usize,
) {
    // Compute the exponentials x - max
    let mut buffer = vec![0_i32; view.len()];
    let shift = 26 - src_fixed_point;
    if src_is_signed {
        // We have to put the signed values in the unsigned range
        let max = view.iter().map(|it| it.wrapping_add(128)).max().unwrap();
        view.iter().zip(buffer.iter_mut()).for_each(|(x, exp)| {
            *exp = exp_on_negative_values((x.wrapping_add(128) as i32 - max as i32) << shift)
        });
    } else {
        let max = view.iter().max().unwrap();
        view.iter().zip(buffer.iter_mut()).for_each(|(x, exp)| {
            let exp_ = exp_on_negative_values((*x as i32 - *max as i32) << shift);
            *exp = exp_;
        });
    }

    // Compute sum of exp and 1/sum of exp
    let sum_of_exp = buffer.iter().map(|it| rescale(*it, 0, 12)).sum();
    let (inv_sum_of_exp, num_bits_over_unit) = get_reciprocal(sum_of_exp, 12);

    // Do the final computation
    view.iter_mut().zip(buffer.iter()).for_each(|(it, exp)| {
        let exponent = num_bits_over_unit + 31 - dst_fixed_point;

        let unsat_output = rounding_divide_by_pot(
            saturating_rounding_doubling_high_mul(inv_sum_of_exp, *exp),
            exponent as i32,
        );

        if dst_is_signed {
            *it = unsafe {
                std::mem::transmute(i32::max(
                    i32::min(unsat_output, i8::max_value() as i32),
                    i8::min_value() as i32,
                ) as i8)
            };
        } else {
            *it = i32::max(i32::min(unsat_output, u8::max_value() as i32), u8::min_value() as i32)
                as u8;
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
        let in_epsilon = 2_f32.powi(-(fixed_point(in_dt).unwrap() as i32));
        let out_epsilon = 2_f32.powi(-(fixed_point(out_dt).unwrap() as i32));
        let epsilon = f32::max(in_epsilon, out_epsilon);
        let error = (found - expected).abs();
        assert!(
            error <= epsilon,
            "epsilon eq failed: |{:?}-{:?}|={} should be <= {}",
            found,
            expected,
            error,
            epsilon
        );
    }

    // Generate a random tensor with a quantized datum type
    fn qtensor<T: PrimInt + Datum + Arbitrary>(shape: Vec<usize>) -> BoxedStrategy<Tensor> {
        let len = shape.iter().product::<usize>();
        let dt = q_datum::<T>();
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
    fn q_datum<T: PrimInt + Datum>() -> BoxedStrategy<DatumType> {
        let max_fixed_point = T::datum_type().size_of() * 8 - T::datum_type().is_signed() as usize;
        (1usize..(max_fixed_point + 1))
            .prop_map(|fixed_point| {
                if T::datum_type().is_signed() {
                    DatumType::QI8(ZpScale {
                        zero_point: 0,
                        scale: 2_f32.powi(-(fixed_point as i32)),
                    })
                } else {
                    DatumType::QU8(ZpScale {
                        zero_point: 0,
                        scale: 2_f32.powi(-(fixed_point as i32)),
                    })
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
            let inputs = tvec!(self.data.clone().into_arc_tensor());
            let softmax = Softmax { axes: self.axes.clone(), output_dt: self.output_dt };

            // Compute quantized output
            let mut result = softmax.eval(inputs)?;
            let result = args_1!(result);
            let result_float = result.cast_to::<f32>()?;

            // Compute reference output
            let input_float = self.data.cast_to::<f32>()?;
            let inputs_float = tvec!(input_float.into_owned().into_arc_tensor());
            let softmax_float = Softmax { axes: self.axes.clone(), output_dt: DatumType::F32 };
            let mut reference_float = softmax_float.eval(inputs_float)?;
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
                        NCHW.from_n_c_hw(n, c, &[h, w]).unwrap().shape.iter().cloned().collect();
                    (
                        prop_oneof![qtensor::<i8>(shape_in.clone()), qtensor::<u8>(shape_in)],
                        Just(tvec![axis]),
                        prop_oneof![q_datum::<u8>(), q_datum::<i8>()],
                    )
                })
                .prop_map(|(data, axes, output_dt)| SoftmaxProblem { data, axes, output_dt })
                .boxed()
        }
    }

    proptest::proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]
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
}
