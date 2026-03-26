use crate::internal::*;
use crate::ops::array::{Slice, TypedConcat};
use crate::ops::binary::TypedBinOp;
use crate::ops::logic::{And, ConditionPattern, Iff, classify_condition_tdim};
use crate::optim::OptimizerSession;

/// Optimizer pass that exploits boolean-valued `uniform_tdim` on wires feeding `Iff` and `Mul`.
///
/// For each such wire it injects concrete constants or restructures the graph:
///
/// - **AllTrue / AllFalse** on an `Iff` condition → shunt output to the matching branch.
/// - **TwoRegions** on an `Iff` condition → `concat(slice(lower_branch), slice(upper_branch))`.
/// - **AllTrue** on a `Mul` operand → inject `Const(1, [1]×rank)`; `declutter_neutral` then
///   folds `Mul(signal, 1) → signal`.
/// - **AllFalse** on a `Mul` operand → inject `Const(0, [1]×rank)`; `declutter_mul` then
///   folds `Mul(signal, 0) → zeros`.
/// - **TwoRegions** on a `Mul` operand → slice the signal, create two sub-`Mul`s each with
///   a concrete constant mask, concat; subsequent iterations fold each sub-`Mul`.
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
    if node.op_is::<Iff>() {
        return try_fold_iff(model, node);
    }
    if let Some(bin_op) = node.op_as::<TypedBinOp>() {
        if bin_op.0.is::<And>()
            && model.outlet_fact(node.id.into())?.datum_type == bool::datum_type()
        {
            return try_fold_bitand_bool(model, node);
        }
        if bin_op.0.neutral_element() == Some(1) {
            return try_fold_mul_mask(model, node);
        }
    }
    Ok(None)
}

// ── Iff ──────────────────────────────────────────────────────────────────────

fn try_fold_iff(model: &TypedModel, node: &TypedNode) -> TractResult<Option<TypedModelPatch>> {
    let cond_fact = model.outlet_fact(node.inputs[0])?;
    let Some(expr) = &cond_fact.uniform_tdim else { return Ok(None) };
    let Some(pattern) = classify_condition_tdim(expr, &cond_fact.shape) else { return Ok(None) };

    match pattern {
        ConditionPattern::AllTrue => shunt_output(model, node, node.inputs[1]),
        ConditionPattern::AllFalse => shunt_output(model, node, node.inputs[2]),
        ConditionPattern::TwoRegions { axis, split, lower_is_true } => {
            let out_fact = model.outlet_fact(node.id.into())?;
            let out_dim = out_fact.shape[axis].clone();
            let (lower_outlet, upper_outlet) = if lower_is_true {
                (node.inputs[1], node.inputs[2])
            } else {
                (node.inputs[2], node.inputs[1])
            };
            // Both branches must span the full output dimension along this axis.
            if model.outlet_fact(node.inputs[1])?.shape[axis] != out_dim
                || model.outlet_fact(node.inputs[2])?.shape[axis] != out_dim
            {
                return Ok(None);
            }
            slice_and_concat(model, node, lower_outlet, upper_outlet, axis, split, out_dim)
        }
    }
}

// ── Mul by boolean mask ───────────────────────────────────────────────────────

fn try_fold_mul_mask(model: &TypedModel, node: &TypedNode) -> TractResult<Option<TypedModelPatch>> {
    let input_facts = model.node_input_facts(node.id)?;

    for mask_ix in 0..2usize {
        let mask_fact = input_facts[mask_ix];
        let Some(expr) = &mask_fact.uniform_tdim else { continue };
        let Some(pattern) = classify_condition_tdim(expr, &mask_fact.shape) else { continue };

        let signal_ix = 1 - mask_ix;
        let mask_dt = mask_fact.datum_type;
        let mask_rank = mask_fact.rank();

        match pattern {
            ConditionPattern::AllTrue => {
                return inject_const_mask(model, node, mask_ix, mask_dt, mask_rank, true);
            }
            ConditionPattern::AllFalse => {
                return inject_const_mask(model, node, mask_ix, mask_dt, mask_rank, false);
            }
            ConditionPattern::TwoRegions { axis, split, lower_is_true } => {
                let signal_outlet = node.inputs[signal_ix];
                let signal_fact = input_facts[signal_ix];
                // Signal must not be broadcast (size-1) along the split axis.
                if signal_fact.shape[axis].is_one() {
                    continue;
                }
                let out_fact = model.outlet_fact(node.id.into())?;
                let out_dim = out_fact.shape[axis].clone();
                if signal_fact.shape[axis] != out_dim {
                    continue;
                }
                if let Some(patch) = split_mul_two_regions(
                    model,
                    node,
                    mask_ix,
                    signal_ix,
                    signal_outlet,
                    mask_dt,
                    mask_rank,
                    axis,
                    split,
                    out_dim,
                    lower_is_true,
                )? {
                    return Ok(Some(patch));
                }
            }
        }
    }
    Ok(None)
}

// ── Bool BitAnd by AllTrue mask ───────────────────────────────────────────────

/// Returns true if the fact is provably all-true (as a Bool tensor).
/// Checks both `uniform_tdim` (symbolic) and `uniform` (concrete uniform value).
fn fact_is_all_true(fact: &TypedFact) -> bool {
    if let Some(expr) = &fact.uniform_tdim {
        if matches!(classify_condition_tdim(expr, &fact.shape), Some(ConditionPattern::AllTrue)) {
            return true;
        }
    }
    if let Some(u) = &fact.uniform {
        if let Ok(v) = u.cast_to_scalar::<bool>() {
            return v;
        }
    }
    false
}

/// Fold `And(AllTrue, signal) → signal` for Bool-typed And nodes.
/// When one input is provably all-true (identity for boolean AND), the node is a no-op.
fn try_fold_bitand_bool(
    model: &TypedModel,
    node: &TypedNode,
) -> TractResult<Option<TypedModelPatch>> {
    let input_facts = model.node_input_facts(node.id)?;
    let out_fact = model.outlet_fact(node.id.into())?;
    for mask_ix in 0..2usize {
        let mask_fact = input_facts[mask_ix];
        if !fact_is_all_true(mask_fact) {
            continue;
        }
        let signal_ix = 1 - mask_ix;
        let signal_outlet = node.inputs[signal_ix];
        let signal_fact = input_facts[signal_ix];
        if signal_fact.shape == out_fact.shape {
            return shunt_output(model, node, signal_outlet);
        }
    }
    Ok(None)
}

/// Replace the mask input with `Const(is_true ? 1 : 0, shape=[1]*rank)` and rewire the node.
/// `declutter_neutral` will then fold Mul(signal, 1) → signal, and
/// `declutter_mul` will fold Mul(signal, 0) → zeros.
fn inject_const_mask(
    model: &TypedModel,
    node: &TypedNode,
    mask_ix: usize,
    mask_dt: DatumType,
    mask_rank: usize,
    is_true: bool,
) -> TractResult<Option<TypedModelPatch>> {
    let const_tensor = uniform_const(mask_dt, mask_rank, is_true)?;
    let mut patch = TypedModelPatch::default();
    let const_wire = patch.wire_node(
        format!("{}.mask_const", node.name),
        crate::ops::konst::Const::new(const_tensor.into_arc_tensor())?,
        &[],
    )?[0];
    let mut new_inputs = tvec![];
    for (ix, &outlet) in node.inputs.iter().enumerate() {
        new_inputs.push(if ix == mask_ix { const_wire } else { patch.tap_model(model, outlet)? });
    }
    let out = patch.wire_node(&node.name, node.op.clone(), &new_inputs)?[0];
    patch.shunt_outside(model, node.id.into(), out)?;
    Ok(Some(patch))
}

/// Slice the signal along `axis`, emit two sub-`Mul`s each with a concrete constant mask,
/// then concat.  Subsequent pass iterations fold each sub-`Mul` via existing declutter rules.
#[allow(clippy::too_many_arguments)]
fn split_mul_two_regions(
    model: &TypedModel,
    node: &TypedNode,
    mask_ix: usize,
    signal_ix: usize,
    signal_outlet: OutletId,
    mask_dt: DatumType,
    mask_rank: usize,
    axis: usize,
    split: TDim,
    out_dim: TDim,
    lower_is_true: bool,
) -> TractResult<Option<TypedModelPatch>> {
    let mut patch = TypedModelPatch::default();
    let signal_wire = patch.tap_model(model, signal_outlet)?;

    let lower_signal = patch.wire_node(
        format!("{}.signal_lower", node.name),
        Slice::new(axis, TDim::Val(0), split.clone()),
        &[signal_wire],
    )?[0];
    let upper_signal = patch.wire_node(
        format!("{}.signal_upper", node.name),
        Slice::new(axis, split, out_dim),
        &[signal_wire],
    )?[0];

    let const_lower = uniform_const(mask_dt, mask_rank, lower_is_true)?;
    let const_upper = uniform_const(mask_dt, mask_rank, !lower_is_true)?;

    let wire_const = |patch: &mut TypedModelPatch, name: &str, t: Tensor| {
        patch
            .wire_node(name, crate::ops::konst::Const::new(t.into_arc_tensor())?, &[])
            .map(|v| v[0])
    };

    let mask_lower = wire_const(&mut patch, &format!("{}.mask_lower", node.name), const_lower)?;
    let mask_upper = wire_const(&mut patch, &format!("{}.mask_upper", node.name), const_upper)?;

    let mut lower_inputs = tvec![OutletId::new(0, 0); 2];
    lower_inputs[signal_ix] = lower_signal;
    lower_inputs[mask_ix] = mask_lower;

    let mut upper_inputs = tvec![OutletId::new(0, 0); 2];
    upper_inputs[signal_ix] = upper_signal;
    upper_inputs[mask_ix] = mask_upper;

    let lower_mul =
        patch.wire_node(format!("{}.lower", node.name), node.op.clone(), &lower_inputs)?[0];
    let upper_mul =
        patch.wire_node(format!("{}.upper", node.name), node.op.clone(), &upper_inputs)?[0];

    let concat = patch.wire_node(
        format!("{}.concat", node.name),
        TypedConcat::new(axis),
        &[lower_mul, upper_mul],
    )?[0];

    patch.shunt_outside(model, node.id.into(), concat)?;
    Ok(Some(patch))
}

// ── Shared helpers ────────────────────────────────────────────────────────────

fn shunt_output(
    model: &TypedModel,
    node: &TypedNode,
    outlet: OutletId,
) -> TractResult<Option<TypedModelPatch>> {
    let mut patch = TypedModelPatch::default();
    let wire = patch.tap_model(model, outlet)?;
    patch.shunt_outside(model, node.id.into(), wire)?;
    Ok(Some(patch))
}

fn slice_and_concat(
    model: &TypedModel,
    node: &TypedNode,
    lower_outlet: OutletId,
    upper_outlet: OutletId,
    axis: usize,
    split: TDim,
    out_dim: TDim,
) -> TractResult<Option<TypedModelPatch>> {
    let mut patch = TypedModelPatch::default();
    let lower_wire = patch.tap_model(model, lower_outlet)?;
    let upper_wire = patch.tap_model(model, upper_outlet)?;
    let lower_slice = patch.wire_node(
        format!("{}.lower_slice", node.name),
        Slice::new(axis, TDim::Val(0), split.clone()),
        &[lower_wire],
    )?[0];
    let upper_slice = patch.wire_node(
        format!("{}.upper_slice", node.name),
        Slice::new(axis, split, out_dim),
        &[upper_wire],
    )?[0];
    let concat = patch.wire_node(
        format!("{}.concat", node.name),
        TypedConcat::new(axis),
        &[lower_slice, upper_slice],
    )?[0];
    patch.shunt_outside(model, node.id.into(), concat)?;
    Ok(Some(patch))
}

/// Build a concrete tensor of shape `[1]*rank` with every element equal to
/// `1` (if `is_true`) or `0` (if not).
fn uniform_const(dt: DatumType, rank: usize, is_true: bool) -> TractResult<Tensor> {
    let shape = vec![1usize; rank];
    if is_true {
        let scalar = match dt {
            DatumType::Bool => tensor0(true),
            _ => tensor0(1i64).cast_to_dt(dt)?.into_owned(),
        };
        scalar.into_shape(&shape)
    } else {
        Tensor::zero_dt(dt, &shape)
    }
}
