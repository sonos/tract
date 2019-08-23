use crate::internal::*;

#[derive(Debug, Clone, new, Default)]
pub struct LayerHardmax {
    axis: isize,
}

impl LayerHardmax {
    fn eval_t<D: Datum + ::num_traits::Float + ::num_traits::FromPrimitive>(
        &self,
        input: Arc<Tensor>,
    ) -> TractResult<TVec<Arc<Tensor>>> {
        let array = input.into_tensor().into_array::<D>()?;
        let shape = array.shape().to_vec();
        let axis =
            if self.axis < 0 { shape.len() as isize + self.axis } else { self.axis } as usize;
        let first_dim: usize = array.shape()[0..axis].iter().product();
        let second_dim: usize = array.len() / first_dim;
        let mut array = array.into_shape((first_dim, second_dim))?;
        array.outer_iter_mut().for_each(|mut layer| {
            let max = layer
                .iter()
                .enumerate()
                .rev()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(b.0.cmp(&a.0)))
                .map(|(ix, _)| ix)
                .unwrap_or(0);
            layer
                .iter_mut()
                .enumerate()
                .for_each(|(ix, r)| *r = D::from_usize((ix == max) as usize).unwrap());
        });
        Ok(tvec!(array.into_shape(shape)?.into_arc_tensor()))
    }
}

impl Op for LayerHardmax {
    fn name(&self) -> Cow<str> {
        "LayerHardmax".into()
    }
}

impl StatelessOp for LayerHardmax {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        dispatch_floatlike!(Self::eval_t(input.datum_type())(self, input))
    }
}

impl InferenceRulesOp for LayerHardmax {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        solver: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        rules(solver, inputs, outputs)
    }

    inference_op_as_op!();
    to_typed!();
}

impl TypedOp for LayerHardmax {
    typed_op_as_op!();

    fn output_facts(&self, inputs: TVec<&NormalizedTensorInfo>) -> TractResult<TVec<NormalizedTensorInfo>> {
        Ok(tvec!(inputs[0].clone()))
    }
}

#[derive(Debug, Clone, new, Default)]
pub struct LayerLogSoftmax {
    axis: isize,
}

impl LayerLogSoftmax {
    fn eval_t<D: Datum + ::num_traits::Float + ::num_traits::FromPrimitive + ::std::iter::Sum>(
        &self,
        input: Arc<Tensor>,
    ) -> TractResult<TVec<Arc<Tensor>>> {
        let array = input.into_tensor().into_array::<D>()?;
        let shape = array.shape().to_vec();
        let axis =
            if self.axis < 0 { shape.len() as isize + self.axis } else { self.axis } as usize;
        let first_dim: usize = array.shape()[0..axis].iter().product();
        let second_dim: usize = array.len() / first_dim;
        let mut array = array.into_shape((first_dim, second_dim))?;
        array.outer_iter_mut().for_each(|mut layer| {
            // https://jamesmccaffrey.wordpress.com/2016/03/04/the-max-trick-when-computing-softmax/
            let max: Option<D> = layer
                .iter()
                .max_by(|a, b| a.partial_cmp(&b).unwrap_or(::std::cmp::Ordering::Equal))
                .cloned();
            layer.mapv_inplace(|x| (x - max.unwrap()).exp());
            let divisor = layer.iter().cloned().sum();
            layer.mapv_inplace(|x| (x / divisor).ln());
        });
        Ok(tvec!(array.into_shape(shape)?.into_arc_tensor()))
    }
}

impl Op for LayerLogSoftmax {
    fn name(&self) -> Cow<str> {
        "LayerLogSoftmax".into()
    }
}

impl StatelessOp for LayerLogSoftmax {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        dispatch_floatlike!(Self::eval_t(input.datum_type())(self, input))
    }
}

impl InferenceRulesOp for LayerLogSoftmax {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        solver: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        rules(solver, inputs, outputs)
    }

    inference_op_as_op!();
    to_typed!();
}

impl TypedOp for LayerLogSoftmax {
    typed_op_as_op!();

    fn output_facts(&self, inputs: TVec<&NormalizedTensorInfo>) -> TractResult<TVec<NormalizedTensorInfo>> {
        Ok(tvec!(inputs[0].clone()))
    }
}

#[derive(Debug, Clone, new, Default)]
pub struct LayerSoftmax {
    axis: isize,
}

impl LayerSoftmax {
    fn eval_t<D: Datum + ::num_traits::Float + ::num_traits::FromPrimitive + ::std::iter::Sum>(
        &self,
        input: Arc<Tensor>,
    ) -> TractResult<TVec<Arc<Tensor>>> {
        let array = input.into_tensor().into_array::<D>()?;
        let shape = array.shape().to_vec();
        let axis =
            if self.axis < 0 { shape.len() as isize + self.axis } else { self.axis } as usize;
        let first_dim: usize = array.shape()[0..axis].iter().product();
        let second_dim: usize = array.len() / first_dim;
        let mut array = array.into_shape((first_dim, second_dim))?;
        array.outer_iter_mut().for_each(|mut layer| {
            // https://jamesmccaffrey.wordpress.com/2016/03/04/the-max-trick-when-computing-softmax/
            let max: Option<D> = layer
                .iter()
                .max_by(|a, b| a.partial_cmp(&b).unwrap_or(::std::cmp::Ordering::Equal))
                .cloned();
            layer.mapv_inplace(|x| (x - max.unwrap()).exp());
            let divisor = layer.iter().cloned().sum();
            layer.mapv_inplace(|x| x / divisor);
        });
        Ok(tvec!(array.into_shape(shape)?.into_arc_tensor()))
    }
}

impl Op for LayerSoftmax {
    fn name(&self) -> Cow<str> {
        "LayerSoftmax".into()
    }
}

impl StatelessOp for LayerSoftmax {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        dispatch_floatlike!(Self::eval_t(input.datum_type())(self, input))
    }
}

impl InferenceRulesOp for LayerSoftmax {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        solver: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        rules(solver, inputs, outputs)
    }

    inference_op_as_op!();
    to_typed!();
}

fn rules<'r, 'p: 'r, 's: 'r>(
    s: &mut Solver<'r>,
    inputs: &'p [TensorProxy],
    outputs: &'p [TensorProxy],
) -> InferenceResult {
    check_output_arity(&outputs, 1)?;
    s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
    s.equals(&outputs[0].rank, &inputs[0].rank)?;
    s.equals(&outputs[0].shape, &inputs[0].shape)?;
    Ok(())
}

impl TypedOp for LayerSoftmax {
    typed_op_as_op!();

    fn output_facts(&self, inputs: TVec<&NormalizedTensorInfo>) -> TractResult<TVec<NormalizedTensorInfo>> {
        Ok(tvec!(inputs[0].clone()))
    }
}

