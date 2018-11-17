use ops::prelude::*;

#[derive(Debug, Clone, new)]
pub struct RmDims {
    pub axes: Vec<usize>,
}

impl RmDims {
    fn compute_shape<D: DimLike>(&self, input: &[D]) -> TVec<D> {
        input
            .iter()
            .enumerate()
            .filter(|(ix, _d)| !self.axes.contains(ix))
            .map(|(_ix, d)| *d)
            .collect()
    }

    /// Evaluates the operation given the input tensors.
    fn eval_t<T: Datum>(&self, input: SharedTensor) -> TractResult<TVec<SharedTensor>> {
        let shape = self.compute_shape(input.shape());
        Ok(tvec![input.to_array::<T>()?.into_shape(&*shape)?.into()])
    }
}

impl Op for RmDims {
    fn name(&self) -> &str {
        "RmDims"
    }

    fn pulsify(&self, mut inputs: TVec<&PulsedTensorFact>) -> TractResult<Vec<PulsifiedOp>> {
        let input = args_1!(inputs);
        let mut fact = input.clone();
        fact.shape = self.compute_shape(&input.shape);
        fact.axis -= self.axes.iter().filter(|&ax| *ax <= input.axis).count();
        Ok(vec![PulsifiedOp::new(Box::new(self.clone()), tvec!(fact))])
    }
}

impl StatelessOp for RmDims {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let input = args_1!(inputs);
        dispatch_datum!(Self::eval_t(input.datum_type())(self, input))
    }
}

impl InferenceRulesOp for RmDims {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p SharedTensorsProxy,
        outputs: &'p SharedTensorsProxy,
    ) -> InferenceResult {
        s.equals(&outputs.len, 1)?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        s.equals(
            &outputs[0].rank,
            (&inputs[0].rank).bex() - self.axes.len() as i32,
        )?;
        s.given(&inputs[0].shape, move |s, shape| {
            let output_shape = self.compute_shape(&shape);
            s.equals(&outputs[0].shape, output_shape)
        })
    }
}
