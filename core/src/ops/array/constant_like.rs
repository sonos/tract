use ndarray::*;
use num::traits::AsPrimitive;
use ops::prelude::*;

#[derive(Debug, Clone, new, Default)]
pub struct ConstantLike {
    value: f32,
}

impl ConstantLike {
    pub fn make<T>(&self, shape: &[usize]) -> TractResult<Tensor>
    where
        T: Datum,
        f32: AsPrimitive<T>,
    {
        Ok(Array::<T, _>::from_elem(shape, self.value.as_()).into())
    }
}

impl Op for ConstantLike {
    fn name(&self) -> &str {
        "ConstantLike"
    }
}

impl StatelessOp for ConstantLike {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: TVec<Tensor>) -> TractResult<TVec<Tensor>> {
        let input = args_1!(inputs);
        Ok(tvec!(dispatch_numbers!(Self::make(input.datum_type())(
            self, input.shape()
        ))?))
    }
}

impl InferenceRulesOp for ConstantLike {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    ) -> InferenceResult {
        s.equals(&inputs.len, 1)?;
        s.equals(&outputs.len, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].rank, &outputs[0].rank)?;
        s.given_2(&inputs[0].shape, &inputs[0].datum_type, move |s, shape, dt| {
            if shape.iter().all(|d| d.to_integer().is_ok()) {
                let shape: Vec<usize> = shape
                    .iter()
                    .map(|d| d.to_integer().unwrap() as usize)
                    .collect();
                let value = dispatch_numbers!(Self::make(dt)(self, &shape))?;
                s.equals(&outputs[0].value, value)?;
            }
            Ok(())
        })
    }
}
