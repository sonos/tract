use crate::internal::*;
use crate::ops::array::{Slice, TypedConcat};
use crate::ops::binary::TypedBinOp;
use crate::ops::logic::{And, Iff, classify_true_range};
use crate::optim::OptimizerSession;

/// Optimizer pass that exploits boolean-valued `uniform_tdim` on wires feeding `Iff` and `Mul`.
///
/// For each such wire it injects concrete constants or restructures the graph:
///
/// - **AllTrue / AllFalse** on any supported op → inject `Const(is_true, [1]×rank)`;
///   subsequent declutter folds the op away (e.g. `Iff::declutter`, `declutter_neutral`).
/// - **TwoRegions** on any supported op → slice all other inputs along the split axis,
///   duplicate the op once per region each with a concrete bool const, concat results.
///
/// The per-op logic is limited to `try_fold_node` (which input indices may carry a bool
/// `uniform_tdim`). All transformation logic in `try_fold_uniform_bool_input` and
/// `split_op_two_regions` is op-agnostic.
#[derive(Clone, Debug, Default)]
pub struct FoldUniformMask(usize);

impl super::TypedPass for FoldUniformMask {
    fn reset(&mut self) -> TractResult<()> {
        self.0 = 0;
        Ok(())
    }

    fn next(
        &mut self,
        _session: &mut OptimizerSession,
        model: &TypedModel,
    ) -> TractResult<Option<TypedModelPatch>> {
        for node in &model.nodes[self.0..] {
            self.0 = node.id + 1;
            if let Some(patch) = try_fold_node(model, node)? {
                return Ok(Some(patch));
            }
        }
        Ok(None)
    }
}

fn try_fold_node(model: &TypedModel, node: &TypedNode) -> TractResult<Option<TypedModelPatch>> {
    let bool_indices: &[usize] = if node.op_is::<Iff>() {
        &[0] // only the condition wire can carry bool uniform_tdim
    } else if let Some(bin_op) = node.op_as::<TypedBinOp>() {
        let out_bool = model.outlet_fact(node.id.into())?.datum_type == bool::datum_type();
        if (bin_op.0.is::<And>() && out_bool) || bin_op.0.neutral_element() == Some(1) {
            &[0, 1]
        } else {
            return Ok(None);
        }
    } else {
        return Ok(None);
    };

    for &bool_ix in bool_indices {
        if let Some(patch) = try_fold_uniform_bool_input(model, node, bool_ix)? {
            return Ok(Some(patch));
        }
    }
    Ok(None)
}

// ── Op-agnostic transformation ────────────────────────────────────────────────

fn try_fold_uniform_bool_input(
    model: &TypedModel,
    node: &TypedNode,
    bool_ix: usize,
) -> TractResult<Option<TypedModelPatch>> {
    let bool_fact = model.outlet_fact(node.inputs[bool_ix])?;
    // If the input is already a concrete constant, the op's own declutter handles folding.
    // Skipping here prevents an infinite loop (inject const → same uniform_tdim → inject again).
    rule_if!(bool_fact.konst.is_none());
    rule_if_some!(expr = &bool_fact.uniform_tdim);
    rule_if_some!(range = classify_true_range(expr, &bool_fact.shape));

    let dt = bool_fact.datum_type;
    let rank = bool_fact.rank();
    if range.is_full() {
        return inject_scalar_const(model, node, bool_ix, dt, rank, true);
    }
    if range.is_empty() {
        return inject_scalar_const(model, node, bool_ix, dt, rank, false);
    }
    split_op_regions(model, node, bool_ix, dt, rank, &range)
}

/// Replace the bool input at `bool_ix` with `Const(is_true ? 1 : 0, [1]*rank)` and rewire.
/// Subsequent declutter folds the op:
/// - `Iff(const_true, x, y) → x` via `Iff::declutter`
/// - `Mul(signal, const_1) → signal` via `declutter_neutral`
/// - `Mul(signal, const_0) → zeros` via `declutter_mul`
/// - `And(signal, const_true) → signal` via `declutter_neutral`
fn inject_scalar_const(
    model: &TypedModel,
    node: &TypedNode,
    bool_ix: usize,
    bool_dt: DatumType,
    bool_rank: usize,
    is_true: bool,
) -> TractResult<Option<TypedModelPatch>> {
    let const_tensor = uniform_const(bool_dt, bool_rank, is_true)?;
    let mut patch = TypedModelPatch::default();
    let const_wire = patch.wire_node(
        format!("{}.bool_const", node.name),
        crate::ops::konst::Const::new(const_tensor.into_arc_tensor())?,
        &[],
    )?[0];
    let mut new_inputs = tvec![];
    for (ix, &outlet) in node.inputs.iter().enumerate() {
        new_inputs.push(if ix == bool_ix { const_wire } else { patch.tap_model(model, outlet)? });
    }
    let out = patch.wire_node(&node.name, node.op.clone(), &new_inputs)?[0];
    patch.shunt_outside(model, node.id.into(), out)?;
    Ok(Some(patch))
}

/// Slice all inputs along `range.axis`, duplicate the op once per region
/// (each with a concrete bool const for the bool input), then concat the outputs.
///
/// Handles two-region (one bound is None) and three-region (both bounds are Some) cases.
/// For each non-bool input:
/// - `shape[axis] == 1` → broadcast dimension; share the same wire across all regions.
/// - `shape[axis] == out_dim` → slice per region.
/// - anything else → bail (`Ok(None)`).
fn split_op_regions(
    model: &TypedModel,
    node: &TypedNode,
    bool_ix: usize,
    bool_dt: DatumType,
    bool_rank: usize,
    range: &crate::ops::logic::TrueRange,
) -> TractResult<Option<TypedModelPatch>> {
    let axis = range.axis;
    let out_dim = model.outlet_fact(node.id.into())?.shape[axis].clone();

    // Build the list of (start, end, is_true) regions.
    let regions: TVec<(TDim, TDim, bool)> = match (&range.start, &range.end) {
        (None, Some(e)) => {
            tvec![(TDim::Val(0), e.clone(), true), (e.clone(), out_dim.clone(), false),]
        }
        (Some(s), None) => {
            tvec![(TDim::Val(0), s.clone(), false), (s.clone(), out_dim.clone(), true),]
        }
        (Some(s), Some(e)) => tvec![
            (TDim::Val(0), s.clone(), false),
            (s.clone(), e.clone(), true),
            (e.clone(), out_dim.clone(), false),
        ],
        _ => return Ok(None), // full or empty — handled by caller
    };

    let mut patch = TypedModelPatch::default();
    let mut region_outs = tvec![];

    for (r_start, r_end, is_true) in &regions {
        let mut region_inputs = tvec![OutletId::new(0, 0); node.inputs.len()];
        for (ix, &outlet) in node.inputs.iter().enumerate() {
            if ix == bool_ix {
                let c = uniform_const(bool_dt, bool_rank, *is_true)?;
                region_inputs[ix] = patch.wire_node(
                    format!("{}.bool_{r_start}", node.name),
                    crate::ops::konst::Const::new(c.into_arc_tensor())?,
                    &[],
                )?[0];
            } else {
                let fact = model.outlet_fact(outlet)?;
                let wire = patch.tap_model(model, outlet)?;
                if fact.shape[axis].is_one() {
                    region_inputs[ix] = wire;
                } else if fact.shape[axis] == out_dim {
                    region_inputs[ix] = patch.wire_node(
                        format!("{}.slice_{r_start}_{ix}", node.name),
                        Slice::new(axis, r_start.clone(), r_end.clone()),
                        &[wire],
                    )?[0];
                } else {
                    return Ok(None);
                }
            }
        }
        region_outs.push(
            patch.wire_node(
                format!("{}.region_{r_start}", node.name),
                node.op.clone(),
                &region_inputs,
            )?[0],
        );
    }

    let result =
        patch.wire_node(format!("{}.concat", node.name), TypedConcat::new(axis), &region_outs)?[0];
    patch.shunt_outside(model, node.id.into(), result)?;
    Ok(Some(patch))
}

// ── Shared helpers ────────────────────────────────────────────────────────────

/// Build a tensor of shape `[1]*rank` with every element equal to `1` (if `is_true`) or `0`.
fn uniform_const(dt: DatumType, rank: usize, is_true: bool) -> TractResult<Tensor> {
    tensor0(is_true as i64).cast_to_dt(dt)?.into_owned().broadcast_into_rank(rank)
}
