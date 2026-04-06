use crate::internal::*;
use crate::prelude::DatumType;
use ndarray::*;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Default)]
pub enum ScatterReduction {
    #[default]
    None,
    Add,
    Mul,
    Min,
    Max,
}

impl ScatterReduction {
    pub fn as_str(&self) -> &'static str {
        match self {
            ScatterReduction::None => "none",
            ScatterReduction::Add => "add",
            ScatterReduction::Mul => "mul",
            ScatterReduction::Min => "min",
            ScatterReduction::Max => "max",
        }
    }

    pub fn parse(s: &str) -> TractResult<Self> {
        Ok(match s {
            "none" => ScatterReduction::None,
            "add" => ScatterReduction::Add,
            "mul" => ScatterReduction::Mul,
            "min" => ScatterReduction::Min,
            "max" => ScatterReduction::Max,
            s => bail!("Unknown scatter reduction: {s}"),
        })
    }
}

#[derive(Debug, Clone, new, Hash, PartialEq, Eq)]
pub struct ScatterNd {
    pub reduction: ScatterReduction,
}

impl Op for ScatterNd {
    fn name(&self) -> StaticName {
        "ScatterNd".into()
    }

    op_as_typed_op!();
}

impl ScatterNd {
    unsafe fn eval_t<T: Datum>(
        data: TValue,
        indices: &ArrayViewD<i64>,
        updates: TValue,
    ) -> TractResult<TValue> {
        let mut data = unsafe { data.into_tensor().into_array_unchecked::<T>() };
        let updates_plain = updates.try_as_plain()?;
        let updates_view = unsafe { updates_plain.to_array_view_unchecked::<T>() };
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
        unsafe { tensor.set_datum_type(updates.datum_type()) };
        Ok(tensor.into_tvalue())
    }

    unsafe fn eval_t_reduce<T: Datum + PartialOrd + std::ops::AddAssign + std::ops::MulAssign>(
        data: TValue,
        indices: &ArrayViewD<i64>,
        updates: TValue,
        reduction: ScatterReduction,
    ) -> TractResult<TValue> {
        let mut data = unsafe { data.into_tensor().into_array_unchecked::<T>() };
        let updates_plain = updates.try_as_plain()?;
        let updates_view = unsafe { updates_plain.to_array_view_unchecked::<T>() };
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
            Zip::from(&mut data).and(&updates).for_each(|d, u| match reduction {
                ScatterReduction::Add => *d += u.clone(),
                ScatterReduction::Mul => *d *= u.clone(),
                ScatterReduction::Min => {
                    if u < d {
                        *d = u.clone()
                    }
                }
                ScatterReduction::Max => {
                    if u > d {
                        *d = u.clone()
                    }
                }
                ScatterReduction::None => unreachable!(),
            });
        }
        let mut tensor = data.into_tensor();
        unsafe { tensor.set_datum_type(updates.datum_type()) };
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
                        data, &indices, updates
                    ))?))
                }
                reduction => Ok(tvec!(dispatch_numbers!(Self::eval_t_reduce(data.datum_type())(
                    data, &indices, updates, reduction
                ))?)),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scatter_nd_i64() {
        let data = tensor2(&[[1i64, 2, 3, 4], [5, 6, 7, 8]]);
        let indices = tensor2(&[[0i64], [1]]);
        let updates = tensor2(&[[9i64, 10, 11, 12], [13, 14, 15, 16]]);
        let result = ScatterNd.eval(tvec!(data.into(), indices.into(), updates.into())).unwrap();
        assert_eq!(*result[0], tensor2(&[[9i64, 10, 11, 12], [13, 14, 15, 16]]));
    }

    #[test]
    fn scatter_nd_tdim_data_and_updates() {
        let data = tensor2(&[[1i64, 2], [3, 4]]).cast_to_dt(DatumType::TDim).unwrap().into_owned();
        let indices = tensor2(&[[0i64, 1]]);
        let updates = tensor1(&[99i64]).cast_to_dt(DatumType::TDim).unwrap().into_owned();
        let result = ScatterNd.eval(tvec!(data.into(), indices.into(), updates.into())).unwrap();
        assert_eq!(result[0].datum_type(), DatumType::I64);
        assert_eq!(result[0].try_as_plain().unwrap().as_slice::<i64>().unwrap(), &[1, 99, 3, 4]);
    }

    #[test]
    fn scatter_nd_tdim_updates_i64_data() {
        let data = tensor2(&[[0i64, 0], [0, 0]]);
        let indices = tensor2(&[[1i64, 0]]);
        let updates = tensor1(&[42i64]).cast_to_dt(DatumType::TDim).unwrap().into_owned();
        let result = ScatterNd.eval(tvec!(data.into(), indices.into(), updates.into())).unwrap();
        assert_eq!(result[0].datum_type(), DatumType::I64);
        assert_eq!(result[0].try_as_plain().unwrap().as_slice::<i64>().unwrap(), &[0, 0, 42, 0]);
    }

    #[test]
    fn scatter_nd_i64_updates_tdim_data() {
        let data = tensor2(&[[1i64, 2], [3, 4]]).cast_to_dt(DatumType::TDim).unwrap().into_owned();
        let indices = tensor2(&[[0i64, 0]]);
        let updates = tensor1(&[77i64]);
        let result = ScatterNd.eval(tvec!(data.into(), indices.into(), updates.into())).unwrap();
        assert_eq!(result[0].datum_type(), DatumType::I64);
        assert_eq!(result[0].try_as_plain().unwrap().as_slice::<i64>().unwrap(), &[77, 2, 3, 4]);
    }
}
