use num_traits::AsPrimitive;

use crate::internal::*;
use crate::infer::*;

#[derive(Debug, Clone, new)]
pub struct Size {
    dt: DatumType,
}

impl Size {
    pub fn coerce_to<T>(size: usize) -> TractResult<Arc<Tensor>>
    where
        T: Copy + Datum,
        usize: AsPrimitive<T>,
    {
        Ok(rctensor0(size.as_()))
    }
}

impl Op for Size {
    fn name(&self) -> Cow<str> {
        "Size".into()
    }

    op_as_typed_op!();
    not_a_pulsed_op!();
}

impl StatelessOp for Size {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
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
        s.equals(&outputs[0].datum_type, self.dt)?;
        s.equals(&outputs[0].rank, 0)?;
        Ok(())
    }

    as_op!();
    to_typed!();
}

impl TypedOp for Size {
    fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(TypedFact::dt_shape(self.dt, [0usize; 0].as_ref())?))
    }

    as_op!();
}
