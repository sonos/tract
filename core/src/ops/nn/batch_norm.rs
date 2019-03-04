use self::super::DataFormat;
use crate::ops::prelude::*;
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
        mut inputs: TVec<SharedTensor>,
    ) -> TractResult<TVec<SharedTensor>>
    where
        f32: AsPrimitive<T>,
    {
        let (x, scale, beta, mean, var) = args_5!(&mut inputs);

        let x = x.to_array::<T>()?;
        let c_axis = self.data_format.shape(x.shape()).c_axis();
        let c_dim = self.data_format.shape(x.shape()).c_dim();

        FixedBatchNorm::new(c_axis, c_dim, scale, beta, mean, var, self.epsilon)?.eval(inputs)
    }
}

impl Op for BatchNorm {
    fn name(&self) -> Cow<str> {
        "BatchNorm".into()
    }

    fn reduce(
        &self,
        inputs: TVec<&TensorFact>,
        _outputs: TVec<&TensorFact>,
        phase: ReductionPhase,
    ) -> TractResult<Option<ReducedOpRewire>> {
        if phase == ReductionPhase::Normalize {
            return Ok(None);
        }

        if let (Some(x_shape), Some(dt), Some(scale), Some(beta), Some(mean), Some(var)) = (
            inputs[0].shape.as_concrete_finite()?,
            inputs[1].datum_type.concretize(),
            inputs[1].concretize(),
            inputs[2].concretize(),
            inputs[3].concretize(),
            inputs[4].concretize(),
        ) {
            let c_axis = self.data_format.shape(&x_shape).c_axis();
            let c_dim = self.data_format.shape(&x_shape).c_dim();

            fn fixed<T>(
                c_axis: usize,
                c_dim: usize,
                scale: SharedTensor,
                beta: SharedTensor,
                mean: SharedTensor,
                var: SharedTensor,
                epsilon: f32,
            ) -> TractResult<Box<Op>>
            where
                T: Datum
                    + ::num_traits::Float
                    + ::num_traits::FromPrimitive
                    + ::ndarray::ScalarOperand,
                f32: AsPrimitive<T>,
            {
                Ok(Box::new(FixedBatchNorm::new(
                    c_axis, c_dim, scale, beta, mean, var, epsilon,
                )?))
            }

            let op = dispatch_floatlike!(fixed(dt)(
                c_axis,
                c_dim,
                scale,
                beta,
                mean,
                var,
                self.epsilon
            ))?;
            return Ok(Some(ReducedOpRewire::unary(op)));
        }
        Ok(None)
    }
}

impl StatelessOp for BatchNorm {
    fn eval(&self, inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
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
            let c = self.data_format.shape(shape).c_dim();
            s.equals(&inputs[1].shape[0], c)
        })?;
        Ok(())
    }
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
        scale: SharedTensor,
        beta: SharedTensor,
        mean: SharedTensor,
        var: SharedTensor,
        epsilon: f32,
    ) -> TractResult<FixedBatchNorm<T>> {
        let scale = scale.to_array::<T>()?.into_shape((c_dim,))?;
        let beta = beta.to_array::<T>()?.into_shape((c_dim,))?;
        let mean = mean.to_array::<T>()?.into_shape((c_dim,))?;
        let var = var.to_array::<T>()?.into_shape((c_dim,))?;

        let denominator = (var + epsilon.as_()).map(|x| x.sqrt());

        let slope = &scale / &denominator;
        let intercept = beta - (mean * scale) / denominator;
        Ok(FixedBatchNorm::<T> {
            c_axis,
            c_dim,
            slope,
            intercept,
        })
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
}

impl<T> StatelessOp for FixedBatchNorm<T>
where
    T: Datum + ::num_traits::Float + ::num_traits::FromPrimitive + ::ndarray::ScalarOperand,
    f32: AsPrimitive<T>,
{
    fn eval(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let x = args_1!(inputs);
        let mut x = x.to_array::<T>()?;
        for c in 0..self.c_dim {
            x.slice_axis_mut(Axis(self.c_axis), (c..=c).into())
                .mapv_inplace(|x| x * self.slope[c] + self.intercept[c]);
        }
        return Ok(tvec!(x.into()));
    }
}

impl<T> InferenceRulesOp for FixedBatchNorm<T>
where
    T: Datum + ::num_traits::Float + ::num_traits::FromPrimitive + ::ndarray::ScalarOperand,
    f32: AsPrimitive<T>,
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
        s.equals(&outputs[0].shape, &inputs[0].shape)?;
        Ok(())
    }
}
