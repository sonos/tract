use ndarray::prelude::*;
use num::cast::AsPrimitive;

use ops::prelude::*;

#[derive(Debug, Clone, new)]
pub struct Size {
    dt: DatumType,
}

impl Size {
    pub fn coerce_to<T>(size: usize) -> TractResult<Value>
    where
        T: Datum,
        usize: AsPrimitive<T>,
    {
        Ok(Tensor::from(arr0(size.as_())).into())
    }
}

impl Op for Size {
    fn name(&self) -> &str {
        "Size"
    }
}

impl StatelessOp for Size {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, inputs: TVec<Value>) -> TractResult<TVec<Value>> {
        let size = inputs[0].shape().iter().product();
        Ok(tvec![dispatch_numbers!(Self::coerce_to(self.dt)(size))?])
    }
}

impl InferenceRulesOp for Size {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    ) -> InferenceResult {
        s.equals(&inputs.len, 1)?;
        s.equals(&outputs.len, 1)?;
        s.equals(&outputs[0].rank, 0)?;
        Ok(())
    }
}
