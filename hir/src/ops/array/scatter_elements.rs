use tract_core::ops::cast::wire_cast;

use crate::infer::*;
use crate::internal::*;

#[derive(Debug, Clone, new, Default, Hash)]
pub struct ScatterElements {
    axis: i64,
}

impl Expansion for ScatterElements {
    fn name(&self) -> StaticName {
        "ScatterElements".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 3)?;
        check_output_arity(outputs, 1)?;

        s.given_2(&inputs[0].datum_type, &inputs[2].datum_type, move |s, input, updates| {
            let super_type: DatumType = DatumType::super_type_for([input, updates])
                .with_context(|| format!("No supertype found for {input:?} and {updates:?}"))?;
            s.equals(&outputs[0].datum_type, super_type)
        })?;
        s.equals(&inputs[0].rank, &inputs[1].rank)?;
        s.equals(&inputs[1].shape, &inputs[2].shape)?;
        s.equals(&outputs[0].shape, &inputs[0].shape)?;
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
        let super_type = if let Some(super_type) = DatumType::super_type_for([
            model.outlet_fact(inputs[0])?.datum_type,
            model.outlet_fact(inputs[2])?.datum_type,
        ]) {
            super_type
        } else {
            bail!("Can not type op");
        };
        let casted = wire_cast(prefix, model, &[inputs[0], inputs[2]], super_type)?;
        model.wire_node(
            prefix,
            tract_core::ops::array::ScatterElements { axis },
            &[casted[0], inputs[1], casted[1]],
        )
    }
}
