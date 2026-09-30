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
        // Prefer the limit's symbolic value when it has one: a length
        // re-derived from an earlier Range through Shape/Gather/Cast chains
        // then yields the *same* symbol instead of a fresh one, so downstream
        // volume checks match exactly.
        let limit = model.outlet_fact(inputs[1])?.uniform_tdim.clone();
        eprintln!(
            "DEBUG Range::wire {} limit_utdim={:?} konst={:?} uniform={:?}",
            prefix,
            limit,
            model.outlet_fact(inputs[1])?.konst.as_ref().map(|t| (t.datum_type(), t.volume())),
            model.outlet_fact(inputs[1])?.uniform.as_ref().map(|t| (t.datum_type(), t.volume()))
        );
        let start = scalar_tdim(model.outlet_fact(inputs[0])?);
        let step = scalar_tdim(model.outlet_fact(inputs[2])?);
        let len = match (limit, start, step) {
            // only a strictly positive integer step keeps the symbolic length
            // exact; anything else falls back to a fresh symbol
            (Some(limit), Some(start), Some(step)) if step.as_i64().is_some_and(|s| s > 0) => {
                (limit - start).divceil(step.as_i64().unwrap() as usize)
            }
            _ => model.symbols.new_with_prefix("range").into(),
        };
        // Core Range yields i64 for TDim inputs (both the konst and the
        // dynamic branch of its output_facts), matching the i64 output our
        // inference rules promise for TDim inputs.
        model.wire_node(prefix, tract_core::ops::array::Range::new(len), &inputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infer::InferenceModelExt;

    // A dynamic-length Range whose limit comes from a cast-to-i64 wire (which we
    // translate as a cast to TDim, see onnx's Cast). The expansion's inferred
    // contract — an i64 output — must survive typed translation and evaluation.
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

#[cfg(test)]
mod uniform_tdim_tests {
    use super::*;
    use crate::infer::InferenceModelExt;

    // The MMS shape chain: one data-dependent Range, its length re-derived
    // through Shape -> Gather -> Cast(to=f32) as the limit of a second Range.
    // The symbolic length must survive the Cast so both Ranges share one
    // symbol and the final Reshape matches exactly.
    #[test]
    fn range_length_survives_cast_chain() -> TractResult<()> {
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
        let range_a = model.wire_node("range_a", expand(Range), &[start, limit[0], step])?[0];

        // Shape of range_a's output materializes the dims as a TDim konst
        let shape = model.wire_node(
            "shape",
            expand(crate::ops::array::Shape::new(DatumType::TDim)),
            &[range_a],
        )?;
        let zero = model.add_const("gather_ix", tensor0(0i64))?;
        let limit_b = model.wire_node(
            "gather",
            expand(crate::ops::array::Gather::new(0)),
            &[shape[0], zero],
        )?;
        let limit_b = model.wire_node(
            "cast_f32",
            tract_core::ops::cast::cast(f32::datum_type()),
            &[limit_b[0]],
        )?;

        let start_f = model.add_const("start_f", tensor0(0f32))?;
        let step_f = model.add_const("step_f", tensor0(1f32))?;
        let range_b = model.wire_node("range_b", expand(Range), &[start_f, limit_b[0], step_f])?;

        // reshape range_a ([len_a]) to [len_b]: one shared symbol means the
        // volumes match exactly
        let target = model.wire_node(
            "target",
            expand(crate::ops::array::Shape::new(DatumType::TDim)),
            &[range_b[0]],
        )?;
        let reshaped = model.wire_node(
            "reshaped",
            expand(crate::ops::array::Reshape::default()),
            &[range_a, target[0]],
        )?;
        model.select_output_outlets(&[reshaped[0]])?;

        let typed = model.into_typed()?;
        let found = tract_core::internal::TypedSimplePlan::new(typed.into_optimized()?)?
            .run(tvec!(tensor0(5i64).into_tvalue()))?;
        assert_eq!(*found[0], tensor1(&[0i64, 1, 2, 3, 4]));
        Ok(())
    }
}

fn scalar_tdim(fact: &TypedFact) -> Option<TDim> {
    for t in [fact.uniform.as_deref(), fact.konst.as_deref()].into_iter().flatten() {
        if t.volume() == 1 {
            if t.datum_type() == TDim::datum_type() {
                return Some(t.try_as_plain_ram().ok()?.to_scalar::<TDim>().ok()?.clone());
            } else if let Ok(d) = t.cast_to::<TDim>() {
                if let Ok(s) = d.try_as_plain_ram() {
                    if let Ok(v) = s.to_scalar::<TDim>() {
                        return Some(v.clone());
                    }
                }
            }
        }
    }
    None
}
