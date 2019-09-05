use crate::internal::*;
use itertools::Itertools;

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

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec!(format!("to shape: {}", self.shape.iter().join("x"))))
    }

    op_as_typed_op!();
}

impl StatelessOp for IntoShape {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        dispatch_datum!(Self::eval_t(input.datum_type())(self, input))
    }
}

impl TypedOp for IntoShape {
    typed_op_as_op!();

    fn output_facts(
        &self,
        inputs: &[&TypedTensorInfo],
    ) -> TractResult<TVec<TypedTensorInfo>> {
        Ok(tvec!(TypedTensorInfo::dt_shape(inputs[0].datum_type, &*self.shape)?))
    }
}
