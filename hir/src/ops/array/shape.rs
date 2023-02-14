use crate::infer::*;
use crate::internal::*;

#[derive(Debug, Clone, new, Hash)]
pub struct Shape {
    pub dt: DatumType,
}


impl Expansion for Shape {
    fn name(&self) -> Cow<str> {
        "Shape".into()
    }


    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 1)?;
        check_output_arity(outputs, 1)?;
        s.equals(&outputs[0].rank, 1)?;
        s.equals(&outputs[0].shape[0], inputs[0].rank.bex().to_dim())?;
        s.equals(&outputs[0].datum_type, self.dt.bex())?;
        s.given(&inputs[0].shape, move |s, shape| {
            let shape = tensor1(&shape);
            if let Ok(shape) = shape.cast_to_dt(self.dt) {
                s.equals(&outputs[0].value, shape.into_owned().into_arc_tensor())?;
            }
            Ok(())
        })
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let shape = tensor1(&model.outlet_fact(inputs[0])?.shape.to_tvec());
        let wire = model.add_const(format!("{prefix}.const"), shape)?;
        model.wire_node(prefix, tract_core::ops::cast::cast(self.dt), &[wire])
    }
}
