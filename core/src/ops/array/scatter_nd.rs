use crate::internal::*;
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
