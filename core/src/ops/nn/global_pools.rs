use crate::internal::*;
use ndarray::prelude::*;

#[derive(Debug, Clone, new, Default)]
pub struct GlobalAvgPool {
    //    data_is_nhwc: bool, // default is nchw (onnx)
}

impl GlobalAvgPool {
    fn eval_t<D: Datum + ::num_traits::Float + ::num_traits::FromPrimitive>(
        &self,
        input: Arc<Tensor>,
    ) -> TractResult<TVec<Arc<Tensor>>> {
        let array = input.to_array_view::<D>()?;
        let n = array.shape()[0];
        let c = array.shape()[1];
        let mut final_shape = array.shape().to_vec();
        for dim in final_shape[2..].iter_mut() {
            *dim = 1;
        }
        let divisor_int = array.len() / (n * c);
        let divisor = D::from(divisor_int).unwrap().recip();
        let result: Tensor = array
            .into_shape(((n * c), divisor_int))?
            .sum_axis(Axis(1))
            .map(|x| *x * divisor)
            .into_shape(final_shape)?
            .into();
        Ok(tvec!(result.into()))
    }
}

impl Op for GlobalAvgPool {
    fn name(&self) -> Cow<str> {
        "GlobalAvgPool".into()
    }
    fn validation(&self) -> Validation {
        Validation::Rounding
    }
    op_as_typed_op!();
    not_a_pulsed_op!();
}

impl StatelessOp for GlobalAvgPool {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        dispatch_floatlike!(Self::eval_t(input.datum_type())(self, input))
    }
}

impl TypedOp for GlobalAvgPool {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        output_facts(inputs)
    }
}

#[derive(Debug, Clone, new, Default)]
pub struct GlobalLpPool {
    p: usize, //    data_is_nhwc: bool, // default is nchw (onnx)
}

impl GlobalLpPool {
    fn eval_t<D: Datum + ::num_traits::Float>(
        &self,
        input: Arc<Tensor>,
    ) -> TractResult<TVec<Arc<Tensor>>> {
        let array = input.to_array_view::<D>()?;
        let n = array.shape()[0];
        let c = array.shape()[1];
        let mut final_shape = array.shape().to_vec();
        for dim in final_shape[2..].iter_mut() {
            *dim = 1;
        }
        let divisor = array.len() / (n * c);
        let input = array.into_shape(((n * c), divisor))?;
        let divisor = D::from(divisor).unwrap().recip();
        let result = if self.p == 1 {
            input.fold_axis(Axis(1), D::zero(), |&a, &b| a + b.abs()).map(|a| *a * divisor)
        } else if self.p == 2 {
            input.fold_axis(Axis(1), D::zero(), |&a, &b| a + b * b).map(|a| a.sqrt() * divisor)
        } else {
            input
                .fold_axis(Axis(1), D::zero(), |&a, &b| a + b.abs().powi(self.p as i32))
                .map(|a| a.powf(D::from(self.p).unwrap().recip()) * divisor)
        };
        Ok(tvec!(result.into_shape(final_shape)?.into_arc_tensor()))
    }
}

impl Op for GlobalLpPool {
    fn name(&self) -> Cow<str> {
        "GlobalLpPool".into()
    }
    fn validation(&self) -> Validation {
        Validation::Rounding
    }
    op_as_typed_op!();
    not_a_pulsed_op!();
}

impl StatelessOp for GlobalLpPool {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        dispatch_floatlike!(Self::eval_t(input.datum_type())(self, input))
    }
}
impl TypedOp for GlobalLpPool {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        output_facts(inputs)
    }
}

#[derive(Debug, Clone, new, Default)]
pub struct GlobalMaxPool {
    //    data_is_nhwc: bool, // default is nchw (onnx)
}

impl GlobalMaxPool {
    fn eval_t<D: Datum + ::num_traits::Float>(
        &self,
        input: Arc<Tensor>,
    ) -> TractResult<TVec<Arc<Tensor>>> {
        let array = input.to_array_view::<D>()?;
        let n = array.shape()[0];
        let c = array.shape()[1];
        let mut final_shape = array.shape().to_vec();
        for dim in final_shape[2..].iter_mut() {
            *dim = 1;
        }
        let divisor = array.len() / (n * c);
        let result: Tensor = array
            .into_shape(((n * c), divisor))?
            .fold_axis(Axis(1), D::min_value(), |a, b| a.max(*b))
            .into_shape(final_shape)?
            .into();
        Ok(tvec!(result.into()))
    }
}

impl Op for GlobalMaxPool {
    fn name(&self) -> Cow<str> {
        "GlobalMaxPool".into()
    }
    op_as_typed_op!();
    not_a_pulsed_op!();
}

impl StatelessOp for GlobalMaxPool {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        dispatch_floatlike!(Self::eval_t(input.datum_type())(self, input))
    }
}

impl TypedOp for GlobalMaxPool {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        output_facts(inputs)
    }
}

fn output_facts(inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
    let mut output = inputs[0].clone();
    for i in 2..output.shape.rank() {
        output.shape.set_dim(i, TDim::from(1))?
    }
    Ok(tvec!(output))
}
