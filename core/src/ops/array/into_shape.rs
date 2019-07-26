use crate::internal::*;

#[derive(Debug, Clone, new, Default)]
pub struct IntoShape<D: DimLike> {
    shape: TVec<D>,
}

impl IntoShape {
    /// Evaluates the operation given the input tensors.
    fn eval_t<T: Datum>(
        &self,
        input: Arc<Tensor>,
    ) -> TractResult<TVec<Arc<Tensor>>> {
        Ok(tvec![input.into_tensor().into_array::<T>()?.into_shape(shape)?.into_arc_tensor()])
    }
}

impl Op for Reshape {
    fn name(&self) -> Cow<str> {
        "IntoShape".into()
    }
}

impl StatelessOp for Reshape {
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let (input, shape) = args_2!(inputs);
        let shape: Vec<isize> =
            shape.cast_to::<i64>()?.to_array_view::<i64>()?.iter().map(|&i| i as isize).collect();
        let oshape = self.compute_shape(input.shape(), &shape)?;
        dispatch_datum!(Self::eval_t(input.datum_type())(self, input, &oshape))
    }
}
