use ndarray::*;
use num_traits::AsPrimitive;
use ops::prelude::*;

#[derive(Debug, Clone, new, Default)]
pub struct ConstantLike {
    value: f32,
}

impl ConstantLike {
    pub fn make<T>(&self, shape: &[usize]) -> TractResult<SharedTensor>
    where
        T: Datum,
        f32: AsPrimitive<T>,
    {
        Ok(Array::<T, _>::from_elem(shape, self.value.as_()).into())
    }
}

impl Op for ConstantLike {
    fn name(&self) -> Cow<str> {
        "ConstantLike".into()
    }
}

impl StatelessOp for ConstantLike {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
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
        inputs: &'p SharedTensorsProxy,
        outputs: &'p SharedTensorsProxy,
    ) -> InferenceResult {
        s.equals(&inputs.len, 1)?;
        s.equals(&outputs.len, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].rank, &outputs[0].rank)?;
        s.equals(&inputs[0].shape, &outputs[0].shape)?;
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

#[derive(Debug, Clone, new, Default)]
pub struct EyeLike {
    dt: Option<DatumType>,
    k: isize
}

impl EyeLike {
    pub fn make<T>(&self, (r,c) : (usize, usize)) -> TractResult<SharedTensor>
    where
        T: Datum + num_traits::One + num_traits::Zero,
        f32: AsPrimitive<T>,
    {
        let mut array = Array2::<T>::zeros((r, c));
        for y in 0..r {
            let x = y as isize + self.k;
            if x >= 0 && x < c as isize {
                array[(y, x as usize)] = T::one()
            }
        }
        Ok(array.into_dyn().into())
    }
}

impl Op for EyeLike {
    fn name(&self) -> Cow<str> {
        "EyeLike".into()
    }
}

impl StatelessOp for EyeLike {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let input = args_1!(inputs);
        Ok(tvec!(dispatch_numbers!(Self::make(input.datum_type())(
            self, (input.shape()[0], input.shape()[1])
        ))?))
    }
}

impl InferenceRulesOp for EyeLike {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p SharedTensorsProxy,
        outputs: &'p SharedTensorsProxy,
    ) -> InferenceResult {
        s.equals(&inputs.len, 1)?;
        s.equals(&outputs.len, 1)?;
        if let Some(dt) = self.dt {
            s.equals(&outputs[0].datum_type, dt)?;
        } else {
            s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        }
        s.equals(&inputs[0].rank, 2)?;
        s.equals(&inputs[0].shape, &outputs[0].shape)?;
        s.given(&inputs[0].shape, move |s, shape| {
            if let (Ok(r),Ok(c)) = (shape[0].to_integer(), shape[1].to_integer()) {
                let shape = (r as usize, c as usize);
                if let Some(dt) = self.dt {
                    let value = dispatch_numbers!(Self::make(dt)(self, shape))?;
                    s.equals(&outputs[0].value, value)?;
                } else {
                    s.given(&inputs[0].datum_type, move |s, dt| {
                        let value = dispatch_numbers!(Self::make(dt)(self, shape))?;
                        s.equals(&outputs[0].value, value)
                    })?;
                }
            }
            Ok(())
        })
    }
}
