use crate::internal::*;
use crate::ops::array::{Slice, TypedConcat};
use crate::ops::binary::TypedBinOp;
use crate::ops::logic::{And, Iff, classify_positive_range};
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
    rule_if_some!(range = classify_positive_range(expr, &bool_fact.shape));

    let dt = bool_fact.datum_type;
    let rank = bool_fact.rank();
    if range.is_full() {
        return inject_scalar_const(model, node, bool_ix, dt, rank, true);
    }
    if range.is_empty() {
        return inject_scalar_const(model, node, bool_ix, dt, rank, false);
    }
    rule_if_some!((split, lower_is_true) = range.two_region_split());
    split_op_two_regions(model, node, bool_ix, dt, rank, range.axis, split, lower_is_true)
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

/// Slice all inputs along `axis`, duplicate the op once per region (each with a concrete
/// bool const for the bool input), then concat the two outputs.
///
/// For each non-bool input:
/// - `shape[axis] == 1` → broadcast dimension; share the same wire for both regions.
/// - `shape[axis] == out_dim` → slice it into lower `[0..split]` and upper `[split..out_dim]`.
/// - anything else → bail (return `Ok(None)`).
#[allow(clippy::too_many_arguments)]
fn split_op_two_regions(
    model: &TypedModel,
    node: &TypedNode,
    bool_ix: usize,
    bool_dt: DatumType,
    bool_rank: usize,
    axis: usize,
    split: TDim,
    lower_is_true: bool,
) -> TractResult<Option<TypedModelPatch>> {
    let out_dim = model.outlet_fact(node.id.into())?.shape[axis].clone();
    let mut patch = TypedModelPatch::default();
    let mut lower_inputs = tvec![OutletId::new(0, 0); node.inputs.len()];
    let mut upper_inputs = tvec![OutletId::new(0, 0); node.inputs.len()];

    for (ix, &outlet) in node.inputs.iter().enumerate() {
        if ix == bool_ix {
            let lo = uniform_const(bool_dt, bool_rank, lower_is_true)?;
            let hi = uniform_const(bool_dt, bool_rank, !lower_is_true)?;
            lower_inputs[ix] = patch.wire_node(
                format!("{}.bool_lower", node.name),
                crate::ops::konst::Const::new(lo.into_arc_tensor())?,
                &[],
            )?[0];
            upper_inputs[ix] = patch.wire_node(
                format!("{}.bool_upper", node.name),
                crate::ops::konst::Const::new(hi.into_arc_tensor())?,
                &[],
            )?[0];
        } else {
            let fact = model.outlet_fact(outlet)?;
            let wire = patch.tap_model(model, outlet)?;
            if fact.shape[axis].is_one() {
                lower_inputs[ix] = wire;
                upper_inputs[ix] = wire;
            } else if fact.shape[axis] == out_dim {
                lower_inputs[ix] = patch.wire_node(
                    format!("{}.lower_{ix}", node.name),
                    Slice::new(axis, TDim::Val(0), split.clone()),
                    &[wire],
                )?[0];
                upper_inputs[ix] = patch.wire_node(
                    format!("{}.upper_{ix}", node.name),
                    Slice::new(axis, split.clone(), out_dim.clone()),
                    &[wire],
                )?[0];
            } else {
                return Ok(None);
            }
        }
    }

    let lower_out =
        patch.wire_node(format!("{}.lower", node.name), node.op.clone(), &lower_inputs)?[0];
    let upper_out =
        patch.wire_node(format!("{}.upper", node.name), node.op.clone(), &upper_inputs)?[0];
    let concat = patch.wire_node(
        format!("{}.concat", node.name),
        TypedConcat::new(axis),
        &[lower_out, upper_out],
    )?[0];
    patch.shunt_outside(model, node.id.into(), concat)?;
    Ok(Some(patch))
}

// ── Shared helpers ────────────────────────────────────────────────────────────

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
