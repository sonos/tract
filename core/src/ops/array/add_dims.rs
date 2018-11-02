use ops::prelude::*;

#[derive(Debug, Clone, new)]
pub struct AddDims {
    pub axes: Vec<usize>,
}

impl AddDims {
    fn compute_shape<D: DimLike>(&self, input: &[D]) -> Vec<D> {
        let mut shape = input.to_vec();
        for &axis in &self.axes {
            shape.insert(axis, D::one())
        }
        shape
    }

    /// Evaluates the operation given the input tensors.
    fn eval_t<T: Datum>(&self, input: Value) -> TractResult<TVec<Value>> {
        let shape = self.compute_shape(input.shape());
        Ok(tvec![input.into_array::<T>()?.into_shape(shape)?.into()])
    }
}

impl Op for AddDims {
    fn name(&self) -> &str {
        "AddDims"
    }

    fn pulsify(&self, mut inputs: TVec<&PulsedTensorFact>) -> TractResult<Vec<PulsifiedOp>> {
        let input = args_1!(inputs);
        let mut fact = input.clone();
        fact.shape = self.compute_shape(&input.shape);
        fact.axis += self.axes.iter().filter(|&ax| *ax <= input.axis).count();
        Ok(vec![PulsifiedOp::new(Box::new(self.clone()), tvec!(fact))])
    }
}

impl StatelessOp for AddDims {
    fn eval(&self, mut inputs: TVec<Value>) -> TractResult<TVec<Value>> {
        let input = args_1!(inputs);
        dispatch_datum!(Self::eval_t(input.datum_type())(self, input))
    }
}

impl InferenceRulesOp for AddDims {
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
            (&inputs[0].rank).bex() + self.axes.len() as i64,
        )?;
        s.given(&inputs[0].shape, move |s, shape| {
            let output_shape = self.compute_shape(&shape);
            s.equals(&outputs[0].shape, output_shape)
        })
    }
}
