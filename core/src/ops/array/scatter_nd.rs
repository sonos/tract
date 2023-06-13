use crate::internal::*;
use ndarray::*;

#[derive(Debug, Clone, new, Hash)]
pub struct ScatterNd;



impl Op for ScatterNd {
    fn name(&self) -> Cow<str> {
        "ScatterNd".into()
    }

    op_as_typed_op!();
}

impl ScatterNd {
    unsafe fn eval_t<T: Datum>(
        &self,
        data: TValue,
        indices: &ArrayViewD<i64>,
        updates: TValue,
    ) -> TractResult<TValue> {
        let mut data = data.into_tensor().into_array_unchecked::<T>();
        let updates_view = updates.to_array_view_unchecked::<T>();
        for coords in tract_ndarray::indices(&indices.shape()[..indices.ndim() - 1]) {
            let mut indices_into_data = indices.view();
            let mut updates = updates_view.view();
            for x in coords.slice() {
                indices_into_data.index_axis_inplace(Axis(0), *x);
                updates.index_axis_inplace(Axis(0), *x);
            }
            let mut data = data.view_mut();
            for x in indices_into_data {
                data.index_axis_inplace(Axis(0), *x as usize);
            }

            data.assign(&updates)
        }
        let mut tensor = data.into_tensor();
        tensor.set_datum_type(updates.datum_type());
        Ok(tensor.into_tvalue())
    }
}

impl TypedOp for ScatterNd {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(inputs[0].datum_type.fact(inputs[0].shape.to_tvec())))
    }
}

impl EvalOp for ScatterNd {
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
