use tract_core::ops::array::Gather;

use crate::infer::*;
use crate::internal::*;

#[derive(Debug, Clone, new, Default, Hash)]
pub struct ArrayFeatureExtractor;

impl Expansion for ArrayFeatureExtractor {
    fn name(&self) -> Cow<str> {
        "ArrayFeatureExtractor".into()
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let last_axis = model.outlet_fact(inputs[0])?.rank() - 1;
        let gather_op = Gather { axis: last_axis, output_type: None };

        model.wire_node(prefix, gather_op, inputs)
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        // Expect two inputs:
        // - X: data to be selected
        // - Y: the indices that'll be applied to the last axis
        check_input_arity(inputs, 2)?;

        // We return one tensor containing the selection
        check_output_arity(outputs, 1)?;

        // Check types
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[1].datum_type, i64::datum_type())?;

        // Check ranks
        s.equals(inputs[0].rank.bex() - 1 + inputs[1].rank.bex(), outputs[0].rank.bex())?;

        // Check shapes
        s.given_2(&inputs[0].shape, &inputs[1].shape, move |s, input_shape, indices_shape| {
            let input_rank = input_shape.len();
            let mut output_shape = tvec![];
            output_shape.extend(input_shape.iter().take(input_rank - 1).cloned());
            output_shape.extend(indices_shape.iter().cloned());
            s.equals(&outputs[0].shape, output_shape)?;
            Ok(())
        })?;
        Ok(())
    }
}
