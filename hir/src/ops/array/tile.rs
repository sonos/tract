use crate::internal::*;

#[derive(Debug, Clone, new, Default, Hash)]
pub struct Tile;

tract_data::impl_dyn_hash!(Tile);

impl Expansion for Tile {
    fn name(&self) -> Cow<str> {
        "Tile".into()
    }

    op_hir!();

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 2)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].rank, &outputs[0].rank)?;
        s.equals(&inputs[1].rank, 1)?;
        s.equals(&inputs[1].shape[0], inputs[0].rank.bex().to_dim())?;
        s.given(&inputs[1].value, move |s, mult| {
            for (ix, &m) in mult.cast_to::<i64>()?.as_slice::<i64>()?.iter().enumerate() {
                s.equals(m * inputs[0].shape[ix].bex(), &outputs[0].shape[ix])?;
            }
            Ok(())
        })?;
        Ok(())
    }

    fn wire(
        &self,
        prefix: &str,
        target: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        if let Some(ref mult) = target.outlet_fact(inputs[1])?.konst {
            let mult: TVec<usize> =
                mult.cast_to::<i64>()?.as_slice::<i64>()?.iter().map(|i| *i as usize).collect();
            target.wire_node(prefix, tract_core::ops::array::Tile::new(mult), &inputs[0..1])
        } else {
            bail!("shape input is variable")
        }
    }
}
