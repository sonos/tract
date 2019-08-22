use crate::internal::*;

#[derive(Debug, Clone, new, Default)]
pub struct IntoShape {
    shape: TVec<usize>,
}

impl IntoShape {
    /// Evaluates the operation given the input tensors.
    fn eval_t<T: Datum>(&self, input: Arc<Tensor>) -> TractResult<TVec<Arc<Tensor>>> {
        Ok(tvec![input
            .into_tensor()
            .into_array::<T>()?
            .into_shape(&*self.shape)?
            .into_arc_tensor()])
    }
}

impl Op for IntoShape {
    fn name(&self) -> Cow<str> {
        "IntoShape".into()
    }

    to_typed!();
}

impl StatelessOp for IntoShape {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        dispatch_datum!(Self::eval_t(inputs[0].datum_type())(self, args_1!(inputs)))
    }
}

impl TypedOp for IntoShape {
    typed_op_as_op!();

    fn output_facts(
        &self,
        inputs: TVec<&NormalizedTensorInfo>,
    ) -> TractResult<TVec<NormalizedTensorInfo>> {
        Ok(tvec!(NormalizedTensorInfo::dt_shape(inputs[0].datum_type, &*self.shape)?))
    }
}
