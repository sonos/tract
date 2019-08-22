use self::super::DataFormat;
use crate::internal::*;
use ndarray::prelude::Array1;
use ndarray::Axis;
use num_traits::AsPrimitive;

#[derive(Debug, Clone, new, Default)]
pub struct BatchNorm {
    data_format: DataFormat,
    epsilon: f32,
    spatial: bool,
}

impl BatchNorm {
    fn eval_t<
        T: Datum + ::num_traits::Float + ::num_traits::FromPrimitive + ::ndarray::ScalarOperand,
    >(
        &self,
        mut inputs: TVec<Arc<Tensor>>,
    ) -> TractResult<TVec<Arc<Tensor>>>
    where
        f32: AsPrimitive<T>,
    {
        let (x, scale, beta, mean, var) = args_5!(&mut inputs);

        let c_axis = self.data_format.shape(x.shape()).c_axis();
        let c_dim = *self.data_format.shape(x.shape()).c_dim();

        FixedBatchNorm::new(c_axis, c_dim, scale, beta, mean, var, self.epsilon)?.eval(tvec!(x))
    }
}

impl Op for BatchNorm {
    fn name(&self) -> Cow<str> {
        "BatchNorm".into()
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let inputs = model.node_input_facts(node.id)?;
        let dt = inputs[1].datum_type;

        if let (Some(x_shape), Some(scale), Some(beta), Some(mean), Some(var)) = (
            inputs[0].shape.as_finite(),
            inputs[1].konst.as_ref(),
            inputs[2].konst.as_ref(),
            inputs[3].konst.as_ref(),
            inputs[4].konst.as_ref(),
        ) {
            let c_axis = self.data_format.shape(&x_shape).c_axis();
            let c_dim = *self.data_format.shape(&x_shape).c_dim();

            fn fixed<T>(
                c_axis: usize,
                c_dim: usize,
                scale: Arc<Tensor>,
                beta: Arc<Tensor>,
                mean: Arc<Tensor>,
                var: Arc<Tensor>,
                epsilon: f32,
            ) -> TractResult<Box<dyn TypedOp>>
            where
                T: Datum
                    + ::num_traits::Float
                    + ::num_traits::FromPrimitive
                    + ::ndarray::ScalarOperand,
                f32: AsPrimitive<T>,
            {
                Ok(Box::new(FixedBatchNorm::new(c_axis, c_dim, scale, beta, mean, var, epsilon)?))
            }

            let op = dispatch_floatlike!(fixed(dt)(
                c_axis,
                c_dim,
                scale.clone(),
                beta.clone(),
                mean.clone(),
                var.clone(),
                self.epsilon
            ))?;
            return Ok(Some(TypedModelPatch::single_unary_op(model, node, op)?));
        }
        Ok(None)
    }

    to_typed!();
}

impl TypedOp for BatchNorm {
    stub_typed_op_as_op!();
}

impl StatelessOp for BatchNorm {
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        dispatch_floatlike!(Self::eval_t(inputs[0].datum_type())(self, inputs))
    }
}

impl InferenceRulesOp for BatchNorm {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 5)?;
        check_output_arity(&outputs, 1)?;
        s.equals_all(wrap!(
            &outputs[0].datum_type,
            &inputs[0].datum_type,
            &inputs[1].datum_type,
            &inputs[2].datum_type,
            &inputs[3].datum_type,
            &inputs[4].datum_type
        ))?;
        s.equals(&inputs[0].shape, &outputs[0].shape)?;
        s.equals_all(wrap!(
            &inputs[1].shape,
            &inputs[2].shape,
            &inputs[3].shape,
            &inputs[4].shape
        ))?;
        s.given(&inputs[0].shape, move |s, shape| {
            let shape = self.data_format.shape(shape);
            s.equals(&inputs[1].shape[0], shape.c_dim())
        })?;
        Ok(())
    }

    inference_op_as_op!();
}

#[derive(Debug, Clone)]
pub struct FixedBatchNorm<T>
where
    T: Datum + ::num_traits::Float + ::num_traits::FromPrimitive + ::ndarray::ScalarOperand,
    f32: AsPrimitive<T>,
{
    c_axis: usize,
    c_dim: usize,
    slope: Array1<T>,
    intercept: Array1<T>,
}

impl<T> FixedBatchNorm<T>
where
    T: Datum + ::num_traits::Float + ::num_traits::FromPrimitive + ::ndarray::ScalarOperand,
    f32: AsPrimitive<T>,
{
    fn new(
        c_axis: usize,
        c_dim: usize,
        scale: Arc<Tensor>,
        beta: Arc<Tensor>,
        mean: Arc<Tensor>,
        var: Arc<Tensor>,
        epsilon: f32,
    ) -> TractResult<FixedBatchNorm<T>> {
        let scale = scale.into_tensor().into_array::<T>()?.into_shape((c_dim,))?;
        let beta = beta.into_tensor().into_array::<T>()?.into_shape((c_dim,))?;
        let mean = mean.into_tensor().into_array::<T>()?.into_shape((c_dim,))?;
        let var = var.into_tensor().into_array::<T>()?.into_shape((c_dim,))?;

        let denominator = (var + epsilon.as_()).map(|x| x.sqrt());

        let slope = &scale / &denominator;
        let intercept = beta - (mean * scale) / denominator;
        Ok(FixedBatchNorm::<T> { c_axis, c_dim, slope, intercept })
    }
}

impl<T> Op for FixedBatchNorm<T>
where
    T: Datum + ::num_traits::Float + ::num_traits::FromPrimitive + ::ndarray::ScalarOperand,
    f32: AsPrimitive<T>,
{
    fn name(&self) -> Cow<str> {
        format!("FixedBatchNorm<{:?}>", T::datum_type()).into()
    }

    fn pulsify(
        &self,
        _source: &NormalizedModel,
        node: &NormalizedNode,
        target: &mut PulsedModel,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        let input = mapping[&node.inputs[0]];
        let fact = target.outlet_fact(input)?.clone();
        let id = target.chain_after(input, &*node.name, self.clone(), tvec!(fact))?;
        Ok(tvec!(OutletId::new(id, 0)))
    }

    to_typed!();
}

impl<T> StatelessOp for FixedBatchNorm<T>
where
    T: Datum + ::num_traits::Float + ::num_traits::FromPrimitive + ::ndarray::ScalarOperand,
    f32: AsPrimitive<T>,
{
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let x = args_1!(inputs);
        let mut x = x.into_tensor().into_array::<T>()?;
        for c in 0..self.c_dim {
            x.slice_axis_mut(Axis(self.c_axis), (c..=c).into())
                .mapv_inplace(|x| x * self.slope[c] + self.intercept[c]);
        }
        return Ok(tvec!(x.into_arc_tensor()));
    }
}

impl<T> TypedOp for FixedBatchNorm<T>
where
    T: Datum + ::num_traits::Float + ::num_traits::FromPrimitive + ::ndarray::ScalarOperand,
    f32: AsPrimitive<T>,
{
    stub_typed_op_as_op!();
}
