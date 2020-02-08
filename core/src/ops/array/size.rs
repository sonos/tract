use crate::internal::*;

#[derive(Debug, Clone, new)]
pub struct Size {
    pub dt: DatumType,
}

impl Op for Size {
    fn name(&self) -> Cow<str> {
        "Size".into()
    }

    op_as_typed_op!();
    not_a_pulsed_op!();
}

impl StatelessOp for Size {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let size = inputs[0].shape().iter().product::<usize>();
        let size = rctensor0(size as i64).cast_to_dt(self.dt)?.into_owned();
        Ok(tvec!(size.into_arc_tensor()))
    }
}

impl TypedOp for Size {
    fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(TypedFact::dt_shape(self.dt, [0usize; 0].as_ref())?))
    }

    as_op!();
}
