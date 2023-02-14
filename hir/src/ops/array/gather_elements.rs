use crate::infer::*;
use crate::internal::*;

#[derive(Debug, Clone, new, Default, Hash)]
pub struct GatherElements {
    axis: i64,
}


impl Expansion for GatherElements {
    fn name(&self) -> Cow<str> {
        "GatherElements".into()
    }


    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 2)?;
        check_output_arity(outputs, 1)?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        s.equals(&inputs[0].rank, &inputs[1].rank)?;
        s.equals(&outputs[0].shape, &inputs[1].shape)?;
        Ok(())
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let input_rank = model.outlet_fact(inputs[0])?.rank();
        let axis = if self.axis < 0 { self.axis + input_rank as i64 } else { self.axis } as usize;
        model.wire_node(prefix, tract_core::ops::array::GatherElements { axis }, inputs)
    }
}
