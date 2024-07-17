use tract_core::ops::cast::wire_cast;

use crate::internal::*;

#[derive(Debug, Default, Clone, new, Hash)]
pub struct Range;

impl Expansion for Range {
    fn name(&self) -> Cow<str> {
        "Range".into()
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 3)?;
        check_output_arity(outputs, 1)?;
        s.given_3(
            &inputs[0].datum_type,
            &inputs[1].datum_type,
            &inputs[2].datum_type,
            move |s, dt0, dt1, dt2| {
                let dt =
                    DatumType::super_type_for([dt0, dt1, dt2]).context("No supertype found")?;
                s.equals(dt, &outputs[0].datum_type)
            },
        )?;
        s.equals(&inputs[0].rank, 0)?;
        s.equals(&inputs[1].rank, 0)?;
        s.equals(&inputs[2].rank, 0)?;
        s.equals(&outputs[0].rank, 1)?;
        s.given_3(&inputs[0].value, &inputs[1].value, &inputs[2].value, move |s, v0, v1, v2| {
            let v0 = v0.cast_to::<TDim>()?;
            let v1 = v1.cast_to::<TDim>()?;
            let v2 = v2.cast_to::<i64>()?;
            let out = (v1.to_scalar::<TDim>()?.clone() - v0.to_scalar::<TDim>()?)
                .divceil(*v2.to_scalar::<i64>()? as _);
            s.equals(&outputs[0].shape[0], out)
        })?;
        Ok(())
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let dt: DatumType = DatumType::super_type_for(
            inputs.iter().map(|o| model.outlet_fact(*o).unwrap().datum_type),
        )
        .context("No supertype for inputs")?;
        let inputs = wire_cast(prefix, model, inputs, dt)?;
        let len = model.symbols.new_with_prefix("range");
        model.wire_node(prefix, tract_core::ops::array::Range::new(len.into()), &inputs)
    }
}
