use crate::internal::*;
use ndarray::*;

#[derive(Debug, Clone, new, Hash)]
pub struct ScatterElements {
    pub axis: usize,
}


impl Op for ScatterElements {
    fn name(&self) -> Cow<str> {
        "ScatterElements".into()
    }

    op_as_typed_op!();
}

impl ScatterElements {
    unsafe fn eval_t<T: Datum>(
        &self,
        data: TValue,
        indices: &ArrayViewD<i64>,
        updates: TValue,
    ) -> TractResult<TValue> {
        let mut data = data.into_tensor().into_array_unchecked::<T>();
        let updates_view = updates.to_array_view_unchecked::<T>();
        for (mut coords, value) in updates_view.indexed_iter() {
            let index = indices[&coords];
            coords[self.axis] =
                if index < 0 { index + data.shape()[self.axis] as i64 } else { index } as usize;
            data[coords] = value.clone()
        }
        let mut tensor = data.into_tensor();
        tensor.set_datum_type(updates.datum_type());
        Ok(tensor.into_tvalue())
    }
}

impl TypedOp for ScatterElements {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(inputs[0].datum_type.fact(inputs[0].shape.clone())))
    }
}

impl EvalOp for ScatterElements {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let (data, indices, updates) = args_3!(inputs);
        let indices = indices.cast_to::<i64>()?;
        let indices = indices.to_array_view::<i64>()?;
        if data.datum_type() != updates.datum_type() {
            bail!(
                "Data and update must be of the same type, got {:?} and {:?}",
                data.datum_type(),
                updates.datum_type()
            );
        }
        unsafe {
            Ok(tvec!(dispatch_datum_by_size!(Self::eval_t(data.datum_type())(
                self, data, &indices, updates
            ))?))
        }
    }
}
