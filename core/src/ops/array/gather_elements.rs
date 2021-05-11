use crate::internal::*;
use ndarray::*;

#[derive(Debug, Clone, new, Hash)]
pub struct GatherElements {
    pub axis: usize,
}
impl_dyn_hash!(GatherElements);

impl Op for GatherElements {
    fn name(&self) -> Cow<str> {
        "GatherElements".into()
    }

    op_core_mir!();
    op_as_typed_op!();
}

impl GatherElements {
    unsafe fn eval_t<T: Datum>(
        &self,
        data: Arc<Tensor>,
        indices: &ArrayViewD<i64>,
    ) -> TractResult<Arc<Tensor>> {
        let data_view = data.to_array_view_unchecked::<T>();
        let output = ArrayD::<T>::from_shape_fn(indices.shape(), |mut coords| {
            let index = indices[&coords];
            coords[self.axis] =
                if index < 0 { index + data_view.shape()[self.axis] as i64 } else { index }
                    as usize;
            data_view[coords].clone()
        });
        let mut tensor = output.into_tensor();
        tensor.set_datum_type(data.datum_type());
        Ok(tensor.into_arc_tensor())
    }
}

impl TypedOp for GatherElements {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(TypedFact::dt_shape(inputs[0].datum_type, &*inputs[1].shape.to_tvec())))
    }
}

impl EvalOp for GatherElements {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let (data, indices) = args_2!(inputs);
        let indices = indices.cast_to::<i64>()?;
        let indices = indices.to_array_view::<i64>()?;
        unsafe {
            Ok(tvec!(dispatch_datum_by_size!(Self::eval_t(data.datum_type())(
                &self, data, &indices
            ))?))
        }
    }
}
