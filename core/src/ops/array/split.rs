use crate::ops::prelude::*;
use ndarray::*;

#[derive(Debug, Clone, new, Default)]
pub struct Split {
    axis: usize,
    outputs: usize,
    split: Option<Vec<usize>>,
}

impl Split {
    fn split_dims<D: DimLike>(&self, input: D) -> TractResult<TVec<D>> {
        if let Some(ref split) = self.split.as_ref() {
            Ok(split.iter().map(|&d| D::from(d)).collect())
        } else {
            Ok(tvec!(input/self.outputs;self. outputs))
        }
    }
    fn eval_t<T: Datum>(&self, input: SharedTensor) -> TractResult<TVec<SharedTensor>> {
        let mut current = 0;
        let input = input.to_array_view::<T>()?;
        println!("shape: {:?}", input.shape());
        Ok(self
            .split_dims(input.shape()[self.axis])?
            .iter()
            .map(|d| {
                println!("axis:{} split {:?}", self.axis, current..current+d);
                let slice = input
                    .slice_axis(Axis(self.axis), (current..current + d).into())
                    .to_owned();
                current += d;
                slice.into()
            })
            .collect())
    }
}

impl Op for Split {
    fn name(&self) -> Cow<str> {
        "Split".into()
    }
}

impl StatelessOp for Split {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let input = args_1!(inputs);
        dispatch_datum!(Self::eval_t(input.datum_type())(self, input))
    }
}

impl InferenceRulesOp for Split {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1)?;
        check_output_arity(&outputs, self.outputs)?;
        (0..self.outputs).try_for_each(|i| {
            s.equals(&inputs[0].datum_type, &outputs[i].datum_type)?;
            s.equals(&inputs[0].rank, &outputs[i].rank)
        })?;
        s.given(&inputs[0].shape, move |s, shape| {
            let dims = self.split_dims(shape[self.axis])?;
            for i in 0..self.outputs {
                let mut shape = shape.clone();
                shape[self.axis] = dims[i];
                s.equals(&outputs[i].shape, shape)?;
            }
            Ok(())
        })?;
        Ok(())
    }
}
