use crate::infer::*;
use crate::internal::*;

#[derive(Debug, Clone, new, Hash)]
pub struct Shape {
    pub dt: DatumType,
}
tract_data::impl_dyn_hash!(Shape);

impl Expansion for Shape {
    fn name(&self) -> Cow<str> {
        "Shape".into()
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
        s.equals(&outputs[0].rank, 1)?;
        s.given(&inputs[0].rank, move |s, r| s.equals(&outputs[0].shape[0], r.to_dim()))?;
        s.given(&outputs[0].shape[0], move |s, r| {
            if let Ok(d) = r.to_i64() {
                s.equals(&inputs[0].rank, d)?;
            }
            Ok(())
        })?;
        s.given(&inputs[0].shape, move |s, shape| {
            if shape.iter().any(|d| d.to_i64().is_err()) {
                s.equals(&outputs[0].datum_type, DatumType::TDim)?;
                let tensor = rctensor1(&*shape);
                s.equals(&outputs[0].value, tensor)
            } else if self.dt == DatumType::I64 {
                s.equals(&outputs[0].datum_type, DatumType::I64)?;
                let tensor = rctensor1(
                    &shape.iter().map(|i| i.to_i64().unwrap()).collect::<Vec<_>>(),
                );
                s.equals(&outputs[0].value, tensor)
            } else {
                s.equals(&outputs[0].datum_type, DatumType::I32)?;
                let tensor = rctensor1(
                    &shape.iter().map(|i| i.to_i64().unwrap() as i32).collect::<Vec<_>>(),
                );
                s.equals(&outputs[0].value, tensor)
            }
        })
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let mut shape = tensor1(&model.outlet_fact(inputs[0])?.shape.to_tvec());
        if let Ok(s) = shape.cast_to_dt(self.dt) {
            shape = s.into_owned();
        };
        let wire = model.add_const(prefix, shape)?;
        Ok(tvec!(wire))
    }
}
