use tract_nnef::tract_core::internal::*;
use tract_nnef::tract_core::ops::binary::BinMiniOp;
use tract_nnef::tract_core::ops::math::{Add, Mul, Neg};

#[derive(Clone, Debug, Hash)]
pub struct BasicRotateHalf;

impl Op for BasicRotateHalf {
    fn name(&self) -> Cow<str> {
        "BasicRotateHalf".to_string().into()
    }
    op_as_typed_op!();
}

impl EvalOp for BasicRotateHalf {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let shape: TVec<_> = input.shape().into();
        let mut tensor = Tensor::zero_dt(input.datum_type(), &shape)?;

        let axis = shape.len() - 1;
        ensure!(shape[axis] % 2 == 0, "BasicRotateHalf possible only if the most inner dimension of the shape {:?} is divible by 2", shape);
        let half = shape[axis] / 2;
        unsafe { tensor.assign_slice_unchecked(0..half, &input, half.., axis) };
        Neg {}.eval_in_place(&mut tensor, None)?;
        unsafe { tensor.assign_slice_unchecked(half.., &input, 0..half, axis) };
        Ok(tvec![tensor.into()])
    }
}

impl TypedOp for BasicRotateHalf {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let dt = inputs[0].datum_type;
        let fact = dt.fact(inputs[0].shape.clone());
        Ok(tvec!(fact))
    }

    as_op!();
}



#[derive(Clone, Debug, Hash)]
pub struct BasicApplyRope;

impl BasicApplyRope {
    pub fn is_supported_dt(dt: DatumType) -> bool {
        matches!(dt, DatumType::F32 | DatumType::F16)
    }
}

impl Op for BasicApplyRope {
    fn name(&self) -> Cow<str> {
        "BasicApplyRope".to_string().into()
    }
    op_as_typed_op!();
}

impl EvalOp for BasicApplyRope {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let (input, cos, sin) = args_3!(inputs);
        let rotated_input = args_1!(BasicRotateHalf.eval(tvec![input.clone()])?);
        let mul_with_cos = Mul.eval(input.clone(), cos, input.datum_type())?;
        let mul_with_sin = Mul.eval(rotated_input, sin, input.datum_type())?;
        let output = Add.eval(mul_with_cos.into(), mul_with_sin.into(), input.datum_type())?;
        Ok(tvec![output.into()])
    }
}

impl TypedOp for BasicApplyRope {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let dt = inputs[0].datum_type;
        let fact = dt.fact(inputs[0].shape.clone());
        Ok(tvec!(fact))
    }

    as_op!();
}

#[cfg(test)]
mod tests {
    use super::*;
    use tract_num_traits::AsPrimitive;
    use tract_num_traits::Zero;
    use tract_nnef::tract_core::ops::math::Neg;

    fn run_test_case<F: Datum + Zero + Copy>(a_shape: &[usize]) -> TractResult<()>
    where
        usize: AsPrimitive<F>,
    {
        let a_len = a_shape.iter().product::<usize>();
        let input = Tensor::from_shape(a_shape, &(0..a_len).map(|f| f.as_()).collect::<Vec<F>>())?;
        let rotated = BasicRotateHalf.eval(tvec![input.clone().into()])?;
        let mut back = args_1!(BasicRotateHalf.eval(rotated)?).into_tensor();
        Neg {}.eval_in_place(&mut back, None)?;
        back.close_enough(&input, Approximation::Close)?;
        Ok(())
    }

    #[test]
    fn test_rotate_half() -> TractResult<()> {
        run_test_case::<f32>(&[2, 2])?;
        run_test_case::<f32>(&[512, 512])?;
        run_test_case::<f32>(&[10, 512, 1024])?;

        Ok(())
    }
}
