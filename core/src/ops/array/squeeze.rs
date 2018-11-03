use ops::prelude::*;

use super::RmDims;

#[derive(Debug, Clone, new, Default)]
pub struct Squeeze {
    axes: Option<Vec<usize>>,
}

impl Squeeze {
    fn compute_shape<D: DimLike>(&self, input: &[D]) -> TractResult<TVec<D>> {
        if let Some(ref axes) = self.axes {
            let mut shape:TVec<D> = input.iter().cloned().collect();
            for &axis in axes.iter().rev() {
                if shape.remove(axis) != D::one() {
                    bail!("Attempt to squeeze an axis which dimension in not one");
                }
            }
            Ok(shape)
        } else {
            Ok(input.iter().cloned().filter(|&d| d != D::one()).collect())
        }
    }

    /// Evaluates the operation given the input tensors.
    fn eval_t<T: Datum>(&self, input: Value) -> TractResult<TVec<Value>> {
        let shape = self.compute_shape(input.shape())?;
        Ok(tvec![input.into_array::<T>()?.into_shape(&*shape)?.into()])
    }
}

impl Op for Squeeze {
    fn name(&self) -> &str {
        "Squeeze"
    }

    fn reduce(
        &self,
        _inputs: TVec<&TensorFact>,
        _outputs: TVec<&TensorFact>,
    ) -> TractResult<Option<ReducedOpRewire>> {
        if let Some(dims) = &self.axes {
            Ok(Some(ReducedOpRewire {
                new_op: Box::new(RmDims::new(dims.clone())),
                rewired: tvec!(0),
            }))
        } else {
            Ok(None)
        }
    }
}

impl StatelessOp for Squeeze {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: TVec<Value>) -> TractResult<TVec<Value>> {
        let input = args_1!(inputs);
        dispatch_datum!(Self::eval_t(input.datum_type())(self, input))
    }
}

impl InferenceRulesOp for Squeeze {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    ) -> InferenceResult {
        s.equals(&outputs.len, 1)?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        if let Some(ref axes) = self.axes {
            s.equals(
                &outputs[0].rank,
                (&inputs[0].rank).bex() - axes.len() as i32,
            )?;
        }
        s.given(&inputs[0].shape, move |s, shape| {
            let output_shape = self.compute_shape(&shape)?;
            s.equals(&outputs[0].shape, output_shape)
        })
    }
}
