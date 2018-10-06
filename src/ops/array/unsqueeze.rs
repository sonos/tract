use analyser::rules::prelude::*;
use ops::prelude::*;

#[derive(Debug, Clone, new)]
pub struct Unsqueeze {
    axes: Vec<usize>,
}

impl Unsqueeze {
    fn compute_shape<D:DimLike>(&self, input: &[D]) -> Vec<D> {
        let mut shape = input.to_vec();
        for &axis in &self.axes {
            shape.insert(axis, D::one())
        }
        shape
    }

    /// Evaluates the operation given the input tensors.
    fn eval_t<T: Datum>(&self, input: Value) -> TfdResult<TVec<Value>> {
        let shape = self.compute_shape(input.shape());
        Ok(tvec![input.into_array::<T>()?.into_shape(shape)?.into()])
    }
}

impl Op for Unsqueeze {
    fn name(&self) -> &str {
        "Unsqueeze"
    }

    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: TVec<Value>) -> TfdResult<TVec<Value>> {
        let input = args_1!(inputs);
        dispatch_datum!(Self::eval_t(input.datum_type())(self, input))
    }
}

impl InferenceRulesOp for Unsqueeze {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    ) -> InferenceResult {
        s.equals(&outputs.len, 1)?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        s.equals(&outputs[0].rank, (&inputs[0].rank).bex() + self.axes.len() as i64)?;
        s.given(&inputs[0].shape, move |s, shape| {
            let output_shape = self.compute_shape(&shape);
            s.equals(&outputs[0].shape, output_shape)
        })
    }
}
