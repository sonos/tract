use crate::ops::prelude::*;

#[derive(Debug, Clone, new, Default)]
pub struct Reshape {}

impl Reshape {
    fn compute_shape<D: DimLike>(&self, input: &[D], shape: &[isize]) -> TractResult<Vec<D>> {
        if shape.iter().all(|d| *d > 0) {
            return Ok(shape.iter().map(|&d| D::from(d as usize)).collect());
        }
        let mut result: Vec<D> = shape
            .iter()
            .zip(input.iter())
            .map(|(&shape, &input)| {
                if shape > 0 {
                    D::from(shape as usize)
                } else {
                    input
                }
            })
            .collect();
        if let Some(minus_one) = shape.iter().position(|d| *d == -1) {
            let prod_input: usize = input
                .iter()
                .try_fold(1, |acc, dim| dim.to_integer().map(|a| a as usize * acc))?;
            let prod_shape: usize = result
                .iter()
                .enumerate()
                .filter(|(ix, _)| *ix != minus_one)
                .try_fold(1, |acc, (_, dim)| {
                    dim.to_integer().map(|a| a as usize * acc)
                })?;
            result[minus_one] = D::from(prod_input / prod_shape);
        }
        Ok(result)
    }

    /// Evaluates the operation given the input tensors.
    fn eval_t<T: Datum>(
        &self,
        input: SharedTensor,
        shape: &[usize],
    ) -> TractResult<TVec<SharedTensor>> {
        Ok(tvec![input.to_array::<T>()?.into_shape(shape)?.into()])
    }
}

impl Op for Reshape {
    fn name(&self) -> Cow<str> {
        "Reshape".into()
    }
}

impl StatelessOp for Reshape {
    fn eval(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let (input, shape) = args_2!(inputs);
        let shape: Vec<isize> = shape
            .cast_to::<i64>()?
            .to_array_view::<i64>()?
            .iter()
            .map(|&i| i as isize)
            .collect();
        let oshape = self.compute_shape(input.shape(), &shape)?;
        dispatch_datum!(Self::eval_t(input.datum_type())(self, input, &oshape))
    }
}

impl InferenceRulesOp for Reshape {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        s.given_2(
            &inputs[0].shape,
            &inputs[1].value,
            move |s, ishape, shape| {
                let shape: Vec<isize> = shape
                    .cast_to::<i64>()?
                    .to_array_view::<i64>()?
                    .iter()
                    .map(|&i| i as isize)
                    .collect();
                let shape = self.compute_shape(&ishape, &shape)?;
                s.equals(&outputs[0].shape, ShapeFact::from(shape))
            },
        )
    }
}
