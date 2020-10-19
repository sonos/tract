use crate::infer::*;
use crate::internal::*;

#[derive(Debug, Clone, new, Hash)]
pub struct Size {
    pub dt: DatumType,
}
tract_data::impl_dyn_hash!(Size);

impl Expansion for Size {
    fn name(&self) -> Cow<str> {
        "Size".into()
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
        s.equals(&outputs[0].datum_type, self.dt)?;
        s.equals(&outputs[0].rank, 0)?;
        Ok(())
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let mut size = tensor0(model.outlet_fact(inputs[0])?.shape.iter().maybe_product()?);
        if let Ok(s) = size.cast_to_dt(self.dt) {
            size = s.into_owned();
        }
        let wire = model.add_const(prefix, size)?;
        Ok(tvec!(wire))
    }
}
