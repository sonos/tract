use tract_core::ops::prelude::*;

#[derive(Debug, Clone, new)]
pub struct Fill {
    dt: DatumType,
}

pub fn fill(pb: &crate::tfpb::node_def::NodeDef) -> TractResult<Box<Op>> {
    let dtype = pb.get_attr_datum_type("T")?;
    Ok(Box::new(Fill::new(dtype)))
}

impl Fill {
    fn eval_t<T:Datum + Copy>(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let (shape, value) = args_2!(inputs);
        let value = *value.to_scalar::<T>()?;
        let shape = shape.cast_to::<i32>()?;
        let shape = shape.to_array_view::<i32>()?;
        let array = ::ndarray::Array::from_elem(
            shape.iter().map(|i| *i as usize).collect::<Vec<usize>>(),
            value,
        );
        Ok(tvec![array.into()])
    }
}

impl Op for Fill {
    fn name(&self) -> Cow<str> {
        "tf.Fill".into()
    }
}



impl StatelessOp for Fill {
    fn eval(&self, inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        dispatch_copy!(Self::eval_t(self.dt)(self, inputs))
    }
}

impl InferenceRulesOp for Fill {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 2)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&outputs[0].datum_type, self.dt)?;
        s.equals(&inputs[1].datum_type, self.dt)?;
        s.equals(&inputs[0].rank, 1)?;
        s.equals(&inputs[1].rank, 0)?;
        s.equals(outputs[0].rank.bex().to_dim(), &inputs[0].shape[0])?;
        s.given(&outputs[0].rank, move |s, rank| {
            for dim in 0..(rank as usize) {
                s.equals(&outputs[0].shape[dim], inputs[0].value[dim].bex().to_dim())?;
            }
            Ok(())
        })
    }
}
