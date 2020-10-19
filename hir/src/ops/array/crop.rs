use crate::infer::*;
use crate::internal::*;

#[derive(Debug, Clone, new, Default, Hash)]
pub struct Crop {
    pub axis: usize,
    pub start: usize,
    pub end: usize,
}

tract_data::impl_dyn_hash!(Crop);

impl Expansion for Crop {
    fn name(&self) -> Cow<str> {
        "Crop".into()
    }

    op_hir!();

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].rank, &outputs[0].rank)?;
        s.given(&inputs[0].rank, move |s, rank| {
            (0..rank as usize).try_for_each(|ax| {
                if self.axis == ax {
                    s.equals(
                        &inputs[0].shape[ax],
                        outputs[0].shape[ax].bex() + self.start.to_dim() + self.end.to_dim(),
                    )
                } else {
                    s.equals(&inputs[0].shape[ax], &outputs[0].shape[ax])
                }
            })
        })?;
        Ok(())
    }

    fn wire(
        &self,
        prefix: &str,
        target: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let len = target.outlet_fact(inputs[0])?.shape[self.axis].clone();
        target.wire_node(
            prefix,
            crate::ops::array::Slice::new(
                self.axis as usize,
                self.start.to_dim(),
                len - self.end.to_dim(),
            ),
            inputs,
        )
    }
}
