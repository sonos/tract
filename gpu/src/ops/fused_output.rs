//! Writing a copy-based op's output in the layout its consumer wants.
//!
//! A copy-based op writes its output through `copy_nd`, which takes the
//! destination's strides, so it can write in any layout those strides describe.
//! [`FusedOutputLayout`] carries the axis-op chain a consumer needs: the op
//! declares the chained fact and asks [`make_fused_output_for_node`] for a view
//! of the arena region to write through, and the chain's nodes disappear --
//! one copy per axis op becoming none.
//!
//! [`fuse_output_axis_op`] moves a chain into its producer. It runs before the
//! input-side `fuse_axis_op` of each backend, which would otherwise wrap the
//! producer and hide it.

use tract_core::internal::*;

use crate::fact::DeviceTypedFactExt;
use crate::rule_ensure;
use crate::tensor::DeviceTensor;
use crate::turn_handler::make_tensor_for_node;

use super::change_axes::GpuAxisOp;

/// An op that can write its output through an axis-op chain's strides. The
/// chain is empty unless [`fuse_output_axis_op`] has moved one in.
pub trait FusedOutputLayout {
    fn out_axis_ops(&self) -> &[GpuAxisOp];
    fn with_out_axis_ops(&self, ops: TVec<GpuAxisOp>) -> Box<dyn TypedOp>;
}

/// The copy-based ops that honour an output chain.
pub fn as_fused_output(op: &dyn TypedOp) -> Option<&dyn FusedOutputLayout> {
    if let Some(op) = op.downcast_ref::<super::pulse::GpuDelay>() {
        return Some(op);
    }
    None
}

fn eval_axis_op(op: &AxisOp, symbols: &SymbolValues) -> AxisOp {
    match op {
        AxisOp::Reshape(skip, from, to) => AxisOp::Reshape(
            *skip,
            from.iter().map(|d| d.eval(symbols)).collect(),
            to.iter().map(|d| d.eval(symbols)).collect(),
        ),
        op => op.clone(),
    }
}

/// The shape `chain` turns `shape` into.
pub fn chain_shape(
    shape: &[usize],
    chain: &[GpuAxisOp],
    symbols: &SymbolValues,
) -> TractResult<TVec<usize>> {
    let mut shape: TVec<usize> = shape.into();
    for op in chain {
        eval_axis_op(&op.inner, symbols).change_shape_array(&mut shape, false)?;
    }
    Ok(shape)
}

fn concrete(dims: &[TDim]) -> TractResult<TVec<usize>> {
    dims.iter().map(|d| Ok(d.to_usize()?)).collect()
}

/// The shape and strides addressing a `chain`-shaped tensor in the axis order
/// its producer writes: `chain` applied backwards, each op's inverse carrying
/// the shape and the strides together.
pub fn unchain_strides(
    shape: &[usize],
    strides: &[isize],
    chain: &[GpuAxisOp],
    symbols: &SymbolValues,
) -> TractResult<(TVec<usize>, TVec<isize>)> {
    let mut shape: TVec<usize> = shape.into();
    let mut strides: TVec<isize> = strides.into();
    for op in chain.iter().rev() {
        match eval_axis_op(&op.inner, symbols) {
            AxisOp::Add(ix) => {
                ensure!(shape[ix] == 1, "Add({ix}) carries extent {}", shape[ix]);
                shape.remove(ix);
                strides.remove(ix);
            }
            AxisOp::Rm(ix) => {
                let stride = strides.get(ix).copied().unwrap_or(1);
                shape.insert(ix, 1);
                strides.insert(ix, stride);
            }
            AxisOp::Move(from, to) => {
                let dim = shape.remove(to);
                shape.insert(from, dim);
                let stride = strides.remove(to);
                strides.insert(from, stride);
            }
            AxisOp::Reshape(skip, from, to) => {
                let (from, to) = (concrete(&from)?, concrete(&to)?);
                let group = &strides[skip..skip + to.len()];
                let inner = *group.last().context("Reshape to no axis at all")?;
                let mut nested = inner;
                for (dim, stride) in to.iter().zip(group).rev() {
                    ensure!(
                        *stride == nested,
                        "Reshaped group is not contiguous, so its split has no strides"
                    );
                    nested *= *dim as isize;
                }
                let mut split: TVec<isize> = tvec!();
                let mut stride = inner;
                for dim in from.iter().rev() {
                    split.insert(0, stride);
                    stride *= *dim as isize;
                }
                let tail: TVec<usize> = shape[skip + to.len()..].into();
                shape.truncate(skip);
                shape.extend(from.iter().copied().chain(tail));
                let tail: TVec<isize> = strides[skip + to.len()..].into();
                strides.truncate(skip);
                strides.extend(split.into_iter().chain(tail));
            }
        }
    }
    Ok((shape, strides))
}

/// The view to write the output through, and the tensor to publish. They are
/// the same tensor when `chain` is empty; otherwise the arena holds the chained
/// layout and the view addresses it in the op's own axis order.
pub fn make_fused_output_for_node(
    ctx: &EvalContext,
    dt: DatumType,
    shape: &[usize],
    chain: &[GpuAxisOp],
) -> TractResult<(DeviceTensor, DeviceTensor)> {
    if chain.is_empty() {
        let output = make_tensor_for_node(ctx, dt, shape)?;
        return Ok((output.clone(), output));
    }
    let chained = chain_shape(shape, chain, ctx.symbols)?;
    let published = make_tensor_for_node(ctx, dt, &chained)?;
    let (unchained, strides) =
        unchain_strides(published.shape(), published.strides(), chain, ctx.symbols)?;
    ensure!(unchained.as_slice() == shape, "Output chain leads to {unchained:?}, not {shape:?}");
    let view = published.reshaped(shape.into())?.restrided(strides)?;
    Ok((view, published))
}

/// Axis 0 is where a laned turn addresses its seats, so the op has to keep it
/// as its own leading axis: a chain reordering it or folding it into another axis
/// cannot be honoured through the strides of one seat.
fn keeps_lane_axis(op: &GpuAxisOp) -> bool {
    match op.inner {
        AxisOp::Add(ix) | AxisOp::Rm(ix) => ix > 0,
        AxisOp::Move(from, to) => from > 0 && to > 0,
        AxisOp::Reshape(skip, _, _) => skip > 0,
    }
}

/// A copy is only worth fusing while it still writes runs: the op's innermost
/// axis has to stay the chain's innermost one, or every element lands at its own
/// stride and the scattered write costs more than the copies it replaces.
fn keeps_runs(rank: usize, chain: &[GpuAxisOp]) -> bool {
    let mut axis = rank - 1;
    let mut rank = rank;
    for op in chain {
        let Some(next) = op.inner.transform_axis(axis) else { return false };
        axis = next;
        rank = match &op.inner {
            AxisOp::Add(_) => rank + 1,
            AxisOp::Rm(_) => rank - 1,
            AxisOp::Move(_, _) => rank,
            AxisOp::Reshape(_, from, to) => rank + to.len() - from.len(),
        };
    }
    axis + 1 == rank
}

/// Move the axis ops between a copy-based producer and its consumer into the
/// producer, which then writes its output in the consumer's layout.
pub fn fuse_output_axis_op(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    _node_name: &str,
    axis_op: &GpuAxisOp,
) -> TractResult<Option<TypedModelPatch>> {
    rule_ensure!(keeps_lane_axis(axis_op));
    rule_if_some!(producer = model.single_prec(node.id)?);
    rule_if_some!(fused = as_fused_output(producer.op.as_ref()));
    rule_ensure!(fused.out_axis_ops().is_empty());
    // An op that may hand its input back writes nothing in that case, so taking a
    // chain would trade the consumer's copy for one of its own.
    rule_ensure!(producer.op.forwards_input().is_none());
    rule_ensure!(model.single_succ(producer.id)?.is_some());

    let mut chain = tvec!(axis_op.clone());
    let mut cursor = node;
    while let Some(next) = model.single_succ(cursor.id)? {
        let Some(next_op) = next.op_as::<GpuAxisOp>().filter(|o| keeps_lane_axis(o)) else {
            break;
        };
        chain.push(next_op.clone());
        cursor = next;
    }

    // A Move is the only axis op that changes the layout, so it is the only one
    // whose copy this saves: the input-side pass folds a bare Add, Rm or Reshape
    // into its consumer as a view, for nothing.
    rule_ensure!(chain.iter().any(|op| matches!(op.inner, AxisOp::Move(..))));

    let rank = model.node_output_facts(producer.id)?[0]
        .as_device_fact()
        .map(|f| f.shape.rank())
        .unwrap_or_else(|| model.node_output_facts(producer.id).unwrap()[0].rank());
    rule_ensure!(keeps_runs(rank, &chain));

    let mut patch = TypedModelPatch::default();
    let inputs = patch.taps(model, &producer.inputs)?;
    let out = patch.wire_node(producer.name.clone(), fused.with_out_axis_ops(chain), &inputs)?;
    patch.shunt_outside(model, cursor.id.into(), out[0])?;
    Ok(Some(patch))
}
