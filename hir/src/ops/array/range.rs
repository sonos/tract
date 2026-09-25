use tract_core::ops::cast::wire_cast;

use crate::internal::*;

#[derive(Debug, Default, Clone, new, Hash, PartialEq, Eq)]
pub struct Range;

impl Expansion for Range {
    fn name(&self) -> StaticName {
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
                if dt.is_tdim() {
                    s.equals(&outputs[0].datum_type, i64::datum_type())
                } else {
                    s.equals(dt, &outputs[0].datum_type)
                }
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
            let out = (v1.try_as_plain_ram()?.to_scalar::<TDim>()?.clone()
                - v0.try_as_plain_ram()?.to_scalar::<TDim>()?)
            .divceil(*v2.try_as_plain_ram()?.to_scalar::<i64>()? as _);
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
        let wires =
            model.wire_node(prefix, tract_core::ops::array::Range::new(len.into()), &inputs)?;
        // Our inference rules promise an i64 output when the super type of the
        // inputs is TDim (as happens when a limit comes from onnx Cast-to-i64,
        // which we translate as a cast to TDim). Core Range honors that when it
        // can compute the length from konst inputs, but its dynamic branch
        // yields a TDim wire. Cast back to i64 so the expansion's outputs match
        // the facts the solver inferred.
        if model.outlet_fact(wires[0])?.datum_type.is_tdim() {
            let name = model.unique_name(format!("{prefix}.cast"));
            return model.wire_node(name, tract_core::ops::cast::cast(i64::datum_type()), &wires);
        }
        Ok(wires)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infer::InferenceModelExt;

    // A dynamic-length Range whose limit comes from a cast-to-i64 wire (which we
    // translate as a cast to TDim, see onnx's Cast). The expansion must keep its
    // inferred contract — an i64 output — even though core Range's dynamic
    // branch produces a TDim wire.
    // Regression test for https://github.com/sonos/tract/issues/2928
    #[test]
    fn tdim_limit_dynamic_range() -> TractResult<()> {
        let mut model = InferenceModel::default();
        let limit =
            model.add_source("limit", InferenceFact::from(i64::datum_type().scalar_fact()))?;
        let limit = model.wire_node(
            "limit.tdim",
            tract_core::ops::cast::cast(DatumType::TDim),
            &[limit],
        )?;
        let start = model.add_const("start", tensor0(0i64))?;
        let step = model.add_const("step", tensor0(1i64))?;
        let range = model.wire_node("range", expand(Range), &[start, limit[0], step])?;
        model.select_output_outlets(&range)?;

        let typed = model.into_typed()?;
        let fact = typed.output_fact(0)?;
        assert_eq!(fact.datum_type, i64::datum_type());

        let plan = tract_core::plan::SimplePlan::new(typed.into_optimized()?)?;
        let found = plan.run(tvec!(tensor0(5i64).into_tvalue()))?;
        assert_eq!(*found[0], tensor1(&[0i64, 1, 2, 3, 4]));
        Ok(())
    }
}
