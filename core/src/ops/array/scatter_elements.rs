use super::scatter_nd::ScatterReduction;
use crate::internal::*;
use crate::prelude::DatumType;
use ndarray::*;

#[derive(Debug, Clone, new, Hash, PartialEq, Eq)]
pub struct ScatterElements {
    pub axis: usize,
    pub reduction: ScatterReduction,
}

impl Op for ScatterElements {
    fn name(&self) -> StaticName {
        "ScatterElements".into()
    }

    op_as_typed_op!();
}

impl ScatterElements {
    unsafe fn eval_t<T: Datum>(
        data: TValue,
        indices: &ArrayViewD<i64>,
        updates: TValue,
        axis: usize,
    ) -> TractResult<TValue> {
        let mut data = unsafe { data.into_tensor().into_array_unchecked::<T>() };
        let updates_plain = updates.try_as_plain()?;
        let updates_view = unsafe { updates_plain.to_array_view_unchecked::<T>() };
        for (mut coords, value) in updates_view.indexed_iter() {
            let index = indices[&coords];
            coords[axis] =
                if index < 0 { index + data.shape()[axis] as i64 } else { index } as usize;
            data[coords] = value.clone()
        }
        let mut tensor = data.into_tensor();
        unsafe { tensor.set_datum_type(updates.datum_type()) };
        Ok(tensor.into_tvalue())
    }

    unsafe fn eval_t_reduce<T: Datum + PartialOrd + std::ops::AddAssign + std::ops::MulAssign>(
        data: TValue,
        indices: &ArrayViewD<i64>,
        updates: TValue,
        axis: usize,
        reduction: ScatterReduction,
    ) -> TractResult<TValue> {
        let mut data = unsafe { data.into_tensor().into_array_unchecked::<T>() };
        let updates_plain = updates.try_as_plain()?;
        let updates_view = unsafe { updates_plain.to_array_view_unchecked::<T>() };
        for (mut coords, value) in updates_view.indexed_iter() {
            let index = indices[&coords];
            coords[axis] =
                if index < 0 { index + data.shape()[axis] as i64 } else { index } as usize;
            let d = &mut data[coords];
            match reduction {
                ScatterReduction::Add => *d += value.clone(),
                ScatterReduction::Mul => *d *= value.clone(),
                ScatterReduction::Min => {
                    if value < d {
                        *d = value.clone()
                    }
                }
                ScatterReduction::Max => {
                    if value > d {
                        *d = value.clone()
                    }
                }
                ScatterReduction::None => unreachable!(),
            }
        }
        let mut tensor = data.into_tensor();
        unsafe { tensor.set_datum_type(updates.datum_type()) };
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
        let indices = indices.to_plain_array_view::<i64>()?;
        let (data, updates) =
            if data.datum_type() == DatumType::TDim || updates.datum_type() == DatumType::TDim {
                let data = if data.datum_type() == DatumType::TDim {
                    data.cast_to::<i64>()?.into_owned().into_tvalue()
                } else {
                    data
                };
                let updates = if updates.datum_type() == DatumType::TDim {
                    updates.cast_to::<i64>()?.into_owned().into_tvalue()
                } else {
                    updates
                };
                (data, updates)
            } else {
                (data, updates)
            };
        if data.datum_type() != updates.datum_type() {
            bail!(
                "Data and update must be of the same type, got {:?} and {:?}",
                data.datum_type(),
                updates.datum_type()
            );
        }
        unsafe {
            match self.reduction {
                ScatterReduction::None => {
                    Ok(tvec!(dispatch_datum_by_size!(Self::eval_t(data.datum_type())(
                        data, &indices, updates, self.axis
                    ))?))
                }
                reduction => Ok(tvec!(dispatch_numbers!(Self::eval_t_reduce(data.datum_type())(
                    data, &indices, updates, self.axis, reduction
                ))?)),
            }
        }
    }
}
