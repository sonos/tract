use analyser::rules::prelude::*;
use ops::prelude::*;

#[derive(Debug, Clone, new)]
pub struct RmDims {
    pub axes: Vec<usize>,
}

impl RmDims {
    fn compute_shape<D: DimLike>(&self, input: &[D]) -> Vec<D> {
        input
            .iter()
            .enumerate()
            .filter(|(ix, _d)| !self.axes.contains(ix))
            .map(|(_ix, d)| *d)
            .collect()
    }

    /// Evaluates the operation given the input tensors.
    fn eval_t<T: Datum>(&self, input: Value) -> TfdResult<TVec<Value>> {
        let shape = self.compute_shape(input.shape());
        Ok(tvec![input.into_array::<T>()?.into_shape(shape)?.into()])
    }
}

impl Op for RmDims {
    fn name(&self) -> &str {
        "RmDims"
    }

    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: TVec<Value>) -> TfdResult<TVec<Value>> {
        let input = args_1!(inputs);
        dispatch_datum!(Self::eval_t(input.datum_type())(self, input))
    }
}

impl InferenceRulesOp for RmDims {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    ) -> InferenceResult {
        s.equals(&outputs.len, 1)?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        s.equals(
            &outputs[0].rank,
            (&inputs[0].rank).bex() - self.axes.len() as i64,
        )?;
        s.given(&inputs[0].shape, move |s, shape| {
            let output_shape = self.compute_shape(&shape);
            s.equals(&outputs[0].shape, output_shape)
        })
    }
}
