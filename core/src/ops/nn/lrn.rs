use crate::ops::prelude::*;
use ndarray::prelude::*;

#[derive(Debug, Clone, new, Default)]
pub struct Lrn {
    alpha: f32,
    beta: f32,
    bias: f32,
    size: usize,
}

impl Lrn {
    fn eval_t<T: Datum + ::num_traits::Float + ::num_traits::FromPrimitive + ::std::iter::Sum>(
        &self,
        input: SharedTensor,
    ) -> TractResult<TVec<SharedTensor>> {
        let input = input.to_array_view::<T>()?;
        let channels = input.shape()[1];
        let output = Array::from_shape_fn(input.shape(), |mut coords| {
            let c = coords[1];
            let x = input[&coords];
            let c_min = (c as isize - ((self.size as isize - 1) / 2)).max(0) as usize;
            let c_max = (c + ((self.size - 1).div_ceil(2))).min(channels - 1);
            let square_sum: T = (c_min..c_max)
                .map(|c| {
                    coords[1] = c;
                    input[&coords].powi(2)
                })
                .sum();
            x / (T::from(self.bias).unwrap()
                + T::from(self.alpha).unwrap() / T::from(self.size).unwrap() * square_sum)
                .powf(T::from(self.beta).unwrap())
        });
        Ok(tvec!(output.into()))
    }
}

impl Op for Lrn {
    fn name(&self) -> Cow<str> {
        "Lrn".into()
    }
}

impl StatelessOp for Lrn {
    fn eval(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let input = args_1!(inputs);
        dispatch_floatlike!(Self::eval_t(input.datum_type())(self, input))
    }
}

impl InferenceRulesOp for Lrn {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].shape, &outputs[0].shape)?;
        Ok(())
    }
}
