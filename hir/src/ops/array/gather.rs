use tract_core::ops::cast::cast;

use crate::infer::*;
use crate::internal::*;

#[derive(Debug, Clone, new, Default, Hash)]
pub struct Gather {
    axis: i64,
}

impl Gather {
    pub fn to_type_op(&self, input_rank: usize) -> tract_core::ops::array::Gather {
        let axis = if self.axis < 0 { self.axis + input_rank as i64 } else { self.axis } as usize;
        tract_core::ops::array::Gather::new(axis)
    }
}

impl Expansion for Gather {
    fn name(&self) -> Cow<str> {
        "Gather".into()
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let input_rank = model.outlet_fact(inputs[0])?.rank();
        let mut inputs: TVec<OutletId> = inputs.into();
        inputs[1] = model.wire_node(
            format!("{prefix}.cast_to_i64"),
            cast(i64::datum_type()),
            &[inputs[1]],
        )?[0];
        model.wire_node(prefix, self.to_type_op(input_rank), &inputs)
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 2)?;
        check_output_arity(outputs, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(inputs[0].rank.bex() - 1 + inputs[1].rank.bex(), outputs[0].rank.bex())?;
        s.given_2(&inputs[0].shape, &inputs[1].shape, move |s, input_shape, indices_shape| {
            let rank = input_shape.len();
            let output_shape =
                self.to_type_op(rank).compute_output_shape(&input_shape, &indices_shape)?;
            s.equals(&outputs[0].shape, output_shape)?;
            Ok(())
        })?;
        Ok(())
    }
}
