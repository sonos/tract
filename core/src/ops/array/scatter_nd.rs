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
        data: &mut Tensor,
        indices: &ArrayViewD<i64>,
        updates: &TValue,
    ) -> TractResult<()> {
        let mut data = unsafe { data.to_array_view_mut_unchecked::<T>() };
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
        Ok(())
    }

    unsafe fn eval_t_reduce<T: Datum + PartialOrd + std::ops::AddAssign + std::ops::MulAssign>(
        data: &mut Tensor,
        indices: &ArrayViewD<i64>,
        updates: &TValue,
        reduction: ScatterReduction,
    ) -> TractResult<()> {
        let mut data = unsafe { data.to_array_view_mut_unchecked::<T>() };
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
        Ok(())
    }
}

/// Locates the single axis along which `tuples` is the row-major enumeration of a
/// contiguous block of `data_shape`, every other axis fully covered.
///
/// `tuples` is the flattened constant index tensor, `data_shape.len()` coordinates
/// per tuple. Returns `(axis, start, len)` of the block, or `None` when the tuples
/// are anything else: the comparison is exact and elementwise, so a match means the
/// scatter writes exactly `data[.., start..start + len, ..]` once, in order.
fn scattered_block(tuples: &[i64], data_shape: &[usize]) -> Option<(usize, usize, usize)> {
    let rank = data_shape.len();
    let count = tuples.len() / rank;
    if count == 0 {
        return None;
    }
    for axis in 0..rank {
        let others: usize =
            data_shape.iter().enumerate().filter(|(ax, _)| *ax != axis).map(|(_, d)| *d).product();
        if others == 0 || !count.is_multiple_of(others) {
            continue;
        }
        let len = count / others;
        let Ok(start) = usize::try_from(tuples[axis]) else { continue };
        if start + len > data_shape[axis] {
            continue;
        }
        let mut block: TVec<usize> = data_shape.into();
        block[axis] = len;
        let canonical = tuples.chunks(rank).enumerate().all(|(pos, tuple)| {
            let mut rest = pos;
            (0..rank).rev().all(|ax| {
                let coord = rest % block[ax];
                rest /= block[ax];
                tuple[ax] == (coord + if ax == axis { start } else { 0 }) as i64
            })
        });
        if canonical {
            return Some((axis, start, len));
        }
    }
    None
}

impl TypedOp for ScatterNd {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(inputs[0].datum_type.fact(inputs[0].shape.to_tvec())))
    }

    /// Rewrites a constant-index block assignment into `Concat(axis, [Slice(data,
    /// 0..start), updates, Slice(data, end..dim)])`, dropping empty slices.
    ///
    /// Fires only when the reduction is `None`, the indices are constant with a last
    /// dimension equal to the data rank, the tuples are exactly the row-major
    /// enumeration of a contiguous block along one axis with every other axis fully
    /// covered, and `updates` matches that block in shape and datum type. Symbolic
    /// shapes are declined: full coverage of an axis cannot be established against
    /// constant indices. `TypedConcat::declutter` and `optim::slice` clean up behind
    /// it.
    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        rule_if!(self.reduction == ScatterReduction::None);
        let (data, indices, updates) = args_3!(model.node_input_facts(node.id)?);
        rule_if_some!(konst = &indices.konst);
        rule_if_some!(data_shape = data.shape.as_concrete());
        rule_if_some!(updates_shape = updates.shape.as_concrete());
        rule_if!(data.is_plain() && updates.is_plain());
        rule_if!(data.datum_type == updates.datum_type);
        rule_if!(konst.rank() >= 2 && konst.is_plain());
        rule_if!(*konst.shape().last().unwrap() == data_shape.len());
        let tuples = konst.cast_to::<i64>()?;
        let tuples = tuples.try_as_plain()?.as_slice::<i64>()?;
        rule_if_some!((axis, start, len) = scattered_block(tuples, data_shape));
        let mut block: TVec<usize> = data_shape.into();
        block[axis] = len;
        rule_if!(updates_shape == &block[..]);

        let mut patch = TypedModelPatch::new("ScatterNd as Slice/Concat");
        let data_tap = patch.tap_model(model, node.inputs[0])?;
        let mut parts = tvec!();
        if start > 0 {
            parts.push(
                patch.wire_node(
                    format!("{}.head", node.name),
                    crate::ops::array::Slice::new(axis, 0, start),
                    &[data_tap],
                )?[0],
            );
        }
        parts.push(patch.tap_model(model, node.inputs[2])?);
        if start + len < data_shape[axis] {
            parts.push(
                patch.wire_node(
                    format!("{}.tail", node.name),
                    crate::ops::array::Slice::new(axis, start + len, data_shape[axis]),
                    &[data_tap],
                )?[0],
            );
        }
        let wire = if parts.len() == 1 {
            parts[0]
        } else {
            patch.wire_node(&node.name, crate::ops::array::TypedConcat::new(axis), &parts)?[0]
        };
        patch.shunt_outside(model, node.id.into(), wire)?;
        Ok(Some(patch))
    }
}

impl EvalOp for ScatterNd {
    fn is_pure_function(&self) -> bool {
        true
    }

    fn eval(&self, _ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
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
        let mut data = data.into_tensor();
        unsafe {
            match self.reduction {
                ScatterReduction::None => dispatch_datum_by_size!(
                    Self::eval_t(data.datum_type())(&mut data, &indices, &updates)
                )?,
                reduction => dispatch_numbers!(Self::eval_t_reduce(data.datum_type())(
                    &mut data, &indices, &updates, reduction
                ))?,
            }
        }
        Ok(tvec!(data.into_tvalue()))
    }
}
