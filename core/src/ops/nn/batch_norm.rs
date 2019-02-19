use self::super::DataFormat;
use crate::ops::prelude::*;
use num_traits::AsPrimitive;

#[derive(Debug, Clone, new, Default)]
pub struct BatchNorm {
    data_format: DataFormat,
    epsilon: f32,
    spatial: bool,
}

impl BatchNorm {
    fn eval_t<T: Datum + ::num_traits::Float + ::num_traits::FromPrimitive>(
        &self,
        mut inputs: TVec<SharedTensor>,
    ) -> TractResult<TVec<SharedTensor>>
    where
        f32: AsPrimitive<T>,
    {
        let (x, scale, beta, mean, var) = args_5!(inputs);
        let mut x = x.to_array::<T>()?;
        let c_axis = self.data_format.shape(x.shape()).c_axis();
        let c_dim = self.data_format.shape(x.shape()).c_dim();
        let scale = scale.to_array::<T>()?.into_shape((c_dim,))?;
        let beta = beta.to_array::<T>()?.into_shape((c_dim,))?;
        let mean = mean.to_array::<T>()?.into_shape((c_dim,))?;
        let var = var.to_array::<T>()?.into_shape((c_dim,))?;
        ::ndarray::indices_of(&x).into_iter().for_each(|coords| {
            let c = coords[c_axis];
            let v = x[&coords];
            let v = (v - mean[c]) / (var[c] + self.epsilon.as_()).sqrt();
            let v = v * scale[c] + beta[c];
            x[&coords] = v;
        });
        return Ok(tvec!(x.into()));
    }
}

impl Op for BatchNorm {
    fn name(&self) -> Cow<str> {
        "BatchNorm".into()
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
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    ) -> InferenceResult {
        s.equals(&inputs.len, 5)?;
        s.equals(&outputs.len, 1)?;
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
