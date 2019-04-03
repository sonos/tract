use ndarray::prelude::*;
use num_traits::AsPrimitive;

use crate::internal::*;

#[derive(Debug, Clone, new)]
pub struct Size {
    dt: DatumType,
}

impl Size {
    pub fn coerce_to<T>(size: usize) -> TractResult<SharedTensor>
    where
        T: Copy + Datum,
        usize: AsPrimitive<T>,
    {
        Ok(Tensor::from(arr0(size.as_())).into())
    }
}

impl Op for Size {
    fn name(&self) -> Cow<str> {
        "Size".into()
    }
}

impl StatelessOp for Size {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let size = inputs[0].shape().iter().product();
        Ok(tvec![dispatch_numbers!(Self::coerce_to(self.dt)(size))?])
    }
}

impl InferenceRulesOp for Size {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&outputs[0].rank, 0)?;
        Ok(())
    }
}
