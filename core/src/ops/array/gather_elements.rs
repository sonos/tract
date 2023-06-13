use crate::internal::*;
use ndarray::*;

#[derive(Debug, Clone, new, Hash)]
pub struct GatherElements {
    pub axis: usize,
}


impl Op for GatherElements {
    fn name(&self) -> Cow<str> {
        "GatherElements".into()
    }

    op_as_typed_op!();
}

impl GatherElements {
    unsafe fn eval_t<T: Datum>(
        &self,
        data: TValue,
        indices: &ArrayViewD<i64>,
    ) -> TractResult<TValue> {
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
        Ok(tensor.into_tvalue())
    }
}

impl TypedOp for GatherElements {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(inputs[0].datum_type.fact(&*inputs[1].shape)))
    }
}

impl EvalOp for GatherElements {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let (data, indices) = args_2!(inputs);
        let indices = indices.cast_to::<i64>()?;
        let indices = indices.to_array_view::<i64>()?;
        unsafe {
            Ok(tvec!(dispatch_datum_by_size!(Self::eval_t(data.datum_type())(
                self, data, &indices
            ))?))
        }
    }
}
