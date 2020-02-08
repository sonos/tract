use crate::internal::*;

#[derive(Debug, Clone, new)]
pub struct Shape {
    pub dt: DatumType,
}

impl Op for Shape {
    fn name(&self) -> Cow<str> {
        "Shape".into()
    }

    op_as_typed_op!();
    not_a_pulsed_op!();
}

impl StatelessOp for Shape {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let shape = rctensor1(&inputs[0].shape().iter().map(|&d| d as i64).collect::<Vec<_>>());
        let shape = shape.cast_to_dt(self.dt)?.into_owned();
        Ok(tvec![shape.into_arc_tensor()])
    }
}

impl TypedOp for Shape {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let shape = inputs[0].shape.iter().collect::<TVec<_>>();
        let mut tensor = tensor1(&*shape);
        if shape.iter().all(|d| d.to_integer().is_ok()) {
            tensor = tensor.cast_to_dt(self.dt)?.into_owned();
        }
        Ok(tvec!(TypedFact::from(tensor)))
    }

    as_op!();
}

