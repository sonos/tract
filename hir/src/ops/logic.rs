use crate::infer::*;
use crate::internal::*;

use tract_core::broadcast::multi_broadcast;
use tract_core::ops::binary::wire_cast;
pub use tract_core::ops::binary::wire_with_rank_broadcast;
pub use tract_core::ops::logic::*;

#[derive(Debug, Clone, Hash)]
pub struct Iff;

impl Expansion for Iff {
    fn name(&self) -> Cow<str> {
        "Iff".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 3)?;
        check_output_arity(outputs, 1)?;
        s.equals(&inputs[0].datum_type, DatumType::Bool)?;
        s.given_2(&inputs[1].datum_type, &inputs[2].datum_type, move |s, a, b| {
            let dt = a
                .common_super_type(b)
                .with_context(|| format!("No super type for {a:?} and {b:?}"))?;
            s.equals(&outputs[0].datum_type, dt)
        })?;
        s.given_3(&inputs[0].shape, &inputs[1].shape, &inputs[2].shape, move |s, c, t, f| {
            let shape = multi_broadcast(&[&c, &t, &f])
                .with_context(|| format!("Incompatible shapes {c:?}, {t:?} and {f:?}"))?;
            s.equals(&outputs[0].shape, shape)
        })?;
        Ok(())
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let dta = model.outlet_fact(inputs[1])?.datum_type;
        let dtb = model.outlet_fact(inputs[2])?.datum_type;
        let dt = dta
            .common_super_type(dtb)
            .with_context(|| format!("No super type for {dta:?} and {dtb:?}"))?;
        let mut casted = wire_cast(prefix, model, &inputs[1..], dt)?;
        casted.insert(0, inputs[0]);
        wire_with_rank_broadcast(prefix, model, tract_core::ops::logic::Iff, &casted)
    }
}
