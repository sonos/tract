use tract_core::internal::*;
use tract_core::ops::array::Slice;

use crate::tensor::{DeviceTensor, DeviceTensorExt};
use crate::utils::facts_to_device_facts;

/// One step of a fused view chain.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ViewStep {
    Slice(Slice),
    Axis(AxisOp),
}

/// A chain of layout ops (`GpuSlice` / `GpuAxisOp`) collapsed into a single
/// strided `copy_nd` dispatch. Each folded op used to be its own device copy
/// (slices and moves are real copies, and even Add/Rm/Reshape run as flat
/// memcpys when they cannot be fused into their consumer); composing the
/// chain into one (offset, strides) view of the source pays one copy for the
/// whole chain. Reshapes that cannot be expressed as a restride of the
/// current view (rare) materialize an intermediate contiguous copy, which
/// degrades to the old per-op behavior at worst.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GpuFusedViewCopy {
    pub steps: TVec<ViewStep>,
}

impl GpuFusedViewCopy {
    fn output_shape(&self, input: &[TDim]) -> TractResult<TVec<TDim>> {
        let mut dims: TVec<TDim> = input.into();
        for step in &self.steps {
            match step {
                ViewStep::Slice(slice) => {
                    ensure!(slice.axis < dims.len());
                    dims[slice.axis] = slice.end.clone() - &slice.start;
                }
                ViewStep::Axis(op) => op.change_shape_array(&mut dims, false)?,
            }
        }
        Ok(dims)
    }
}

/// Restrides `old` (shape, strides) into `new_shape` without moving bytes.
/// Returns None when the grouping of dims is not contiguous in the view.
fn try_reshape_strides(
    old_shape: &[usize],
    old_strides: &[isize],
    new_shape: &[usize],
) -> Option<TVec<isize>> {
    if old_shape.iter().any(|&d| d == 0) || new_shape.iter().any(|&d| d == 0) {
        return None;
    }
    let mut new_strides = vec![0isize; new_shape.len()];
    let (mut oi, mut ni) = (0usize, 0usize);
    while oi < old_shape.len() && ni < new_shape.len() {
        let (o0, n0) = (oi, ni);
        let mut op = old_shape[oi] as i64;
        let mut np = new_shape[ni] as i64;
        while op != np {
            if op < np {
                oi += 1;
                op *= *old_shape.get(oi)? as i64;
            } else {
                ni += 1;
                np *= *new_shape.get(ni)? as i64;
            }
        }
        // The old group must be contiguous within the view (size-1 dims
        // don't constrain anything).
        for k in o0..oi {
            if old_shape[k] != 1
                && old_shape[k + 1..=oi].iter().product::<usize>() != 1
                && old_strides[k]
                    != old_strides[k + 1] * old_shape[k + 1] as isize
            {
                return None;
            }
        }
        let mut stride = old_strides[oi];
        for k in (n0..=ni).rev() {
            new_strides[k] = stride;
            stride *= new_shape[k] as isize;
        }
        oi += 1;
        ni += 1;
    }
    while ni < new_shape.len() {
        if new_shape[ni] != 1 {
            return None;
        }
        new_strides[ni] = 1;
        ni += 1;
    }
    while oi < old_shape.len() {
        if old_shape[oi] != 1 {
            return None;
        }
        oi += 1;
    }
    Some(new_strides.into())
}

/// Running view over a base tensor: shape, element strides, element offset.
struct View {
    base: DeviceTensor,
    shape: TVec<usize>,
    strides: TVec<isize>,
    offset: usize,
}

impl View {
    fn contiguous(base: DeviceTensor) -> Self {
        let shape: TVec<usize> = base.shape().into();
        let strides = Tensor::natural_strides(&shape);
        View { base, shape, strides, offset: 0 }
    }

    /// Copies the current view into a fresh contiguous tensor (used both for
    /// the final output and as a fallback when a reshape cannot restride).
    fn materialize(&self, output: &DeviceTensor) -> TractResult<()> {
        let ctx = crate::device::get_context()?;
        let in_strides = crate::utils::compute_broadcast_strides(&self.shape, &self.strides)?;
        ctx.copy_nd(
            &self.base,
            self.offset * self.base.datum_type().size_of(),
            &in_strides,
            output,
            0,
            &self.shape,
            output.strides(),
        )
    }
}

impl Op for GpuFusedViewCopy {
    fn name(&self) -> StaticName {
        "GpuFusedViewCopy".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("steps: {:?}", self.steps)])
    }

    op_as_typed_op!();
}

impl EvalOp for GpuFusedViewCopy {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        node_id: usize,
        session: &TurnState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let input_value = args_1!(inputs);
        let input = input_value.to_device_tensor()?;
        let dt = input.datum_type();
        let mut view = View::contiguous(input.clone());

        for step in &self.steps {
            match step {
                ViewStep::Slice(slice) => {
                    let start = slice.start.eval(&session.resolved_symbols).to_usize()?;
                    let end = slice.end.eval(&session.resolved_symbols).to_usize()?;
                    let axis = slice.axis;
                    ensure!(
                        axis < view.shape.len() && start <= end && end <= view.shape[axis],
                        "invalid slice {start}..{end} on axis {axis} of {:?}",
                        view.shape
                    );
                    view.offset += start * view.strides[axis] as usize;
                    view.shape[axis] = end - start;
                }
                ViewStep::Axis(AxisOp::Move(from, to)) => {
                    let d = view.shape.remove(*from);
                    view.shape.insert(*to, d);
                    let s = view.strides.remove(*from);
                    view.strides.insert(*to, s);
                }
                ViewStep::Axis(AxisOp::Add(axis)) => {
                    view.shape.insert(*axis, 1);
                    view.strides.insert(*axis, 0);
                }
                ViewStep::Axis(AxisOp::Rm(axis)) => {
                    ensure!(view.shape[*axis] == 1);
                    view.shape.remove(*axis);
                    view.strides.remove(*axis);
                }
                ViewStep::Axis(op @ AxisOp::Reshape(..)) => {
                    let AxisOp::Reshape(skip, from, to) = op else { unreachable!() };
                    let from: TVec<TDim> =
                        from.iter().map(|d| d.eval(&session.resolved_symbols)).collect();
                    let to: TVec<TDim> =
                        to.iter().map(|d| d.eval(&session.resolved_symbols)).collect();
                    let mut new_shape = view.shape.clone();
                    AxisOp::Reshape(*skip, from, to).change_shape_array(&mut new_shape, false)?;
                    match try_reshape_strides(&view.shape, &view.strides, &new_shape) {
                        Some(strides) => {
                            view.shape = new_shape;
                            view.strides = strides;
                        }
                        None => {
                            // The view cannot express this reshape: pay one
                            // intermediate contiguous copy and continue.
                            let tmp = DeviceTensor::uninitialized_dt(dt, &view.shape)?;
                            view.materialize(&tmp)?;
                            view = View::contiguous(tmp);
                            view.shape = new_shape.clone();
                            view.strides = Tensor::natural_strides(&new_shape);
                        }
                    }
                }
            }
        }

        // The composed view may be a plain packed region at runtime (e.g. a
        // channel-range slice plus reshapes once the sequence dim is 1 at
        // decode): no copy needed then, just alias the source buffer.
        if view.shape.iter().all(|&d| d != 0)
            && let Some(aliased) = view.base.try_dense_alias(
                &view.shape,
                &view.strides,
                view.offset * dt.size_of(),
            )?
        {
            return Ok(tvec![aliased.into_tensor().into_tvalue()]);
        }

        let output = crate::session_handler::make_tensor_for_node(
            session,
            node_id,
            dt,
            &view.shape,
        )?;
        view.materialize(&output)?;
        Ok(tvec![output.into_tensor().into_tvalue()])
    }
}

impl TypedOp for GpuFusedViewCopy {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        facts_to_device_facts(inputs, |facts| {
            let dims = self.output_shape(&facts[0].shape.to_tvec())?;
            Ok(tvec![facts[0].datum_type.fact(dims)])
        })
        .with_context(|| format!("Error while computing output facts for {}", self.name()))
    }

    as_op!();
}
