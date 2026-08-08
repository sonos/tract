use tract_data::TooEarly;

use crate::internal::*;
use crate::ops::array::Slice;
use crate::ops::dummy::Dummy;
use crate::ops::konst::{Const, LazyConst};
use crate::ops::source::TypedSource;
use crate::optim::{CONST_FOLD_MEM_BUDGET, OptimizerSession};

/// Replaces stateless nodes whose inputs are all constant with the constant they
/// evaluate to, walking forward through single-successor chains so one patch can
/// collapse a whole run. Folds are bounded by [`CONST_FOLD_MEM_BUDGET`] so a large
/// weight is not duplicated once per consumer.
#[derive(Clone, Debug, Default)]
pub struct PropConst(usize);

impl super::TypedPass for PropConst {
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
            if node.op_is::<Const>() && node.outputs[0].fact.konst.is_none() {
                self.0 = node.id;
                let mut patch = TypedModelPatch::default();
                let wire =
                    patch.add_const(&node.name, node.op_as::<Const>().unwrap().val().clone())?;
                patch.shunt_outside(model, node.id.into(), wire)?;
                return Ok(Some(patch));
            }
            let inputs = model.node_input_facts(node.id)?;
            // A LazyConst has no inputs, so "all inputs are constant" holds vacuously; it
            // is a constant awaiting its value, not a computation to fold.
            if !node.op_is::<Const>()
                && !node.op_is::<LazyConst>()
                && !node.op_is::<Dummy>()
                && !node.op_is::<TypedSource>()
                && node.op.is_stateless()
                && inputs.iter().zip(&node.inputs).all(|(fact, outlet)| {
                    fact.konst.is_some()
                        && (model.node(outlet.node).outputs[outlet.slot].successors.len() == 1
                            || node.op_is::<Slice>()
                            || (fact.datum_type.is_number()
                                && fact
                                    .mem_size()
                                    .as_i64()
                                    .is_some_and(|m| m as u64 <= CONST_FOLD_MEM_BUDGET)))
                })
            {
                let inputs =
                    inputs.iter().map(|f| f.konst.clone().unwrap().into_tvalue()).collect();
                let input_mem: u64 = model
                    .node_input_facts(node.id)?
                    .iter()
                    .map(|f| f.mem_size().as_i64().unwrap_or(i64::MAX) as u64)
                    .sum();
                match node.op.eval_with_session(node.id, &TurnState::default(), inputs) {
                    Ok(mut res) => {
                        self.0 = node.id;
                        let output_mem: u64 = res
                            .iter()
                            .map(|t| (t.datum_type().size_of() * t.volume()) as u64)
                            .sum();
                        if output_mem > input_mem.max(CONST_FOLD_MEM_BUDGET) {
                            continue;
                        }
                        let mut node = node;
                        loop {
                            let Some(succ) = model.single_succ(node.id)? else {
                                break;
                            };
                            if succ.inputs.len() > 1 || !succ.op.is_stateless() {
                                break;
                            }
                            let Ok(succ_res) = succ.op.eval_with_session(
                                node.id,
                                &TurnState::default(),
                                res.clone(),
                            ) else {
                                break;
                            };
                            let succ_mem: u64 = succ_res
                                .iter()
                                .map(|t| (t.datum_type().size_of() * t.volume()) as u64)
                                .sum();
                            if succ_mem > input_mem.max(CONST_FOLD_MEM_BUDGET) {
                                break;
                            }
                            res = succ_res;
                            node = succ;
                        }
                        let mut patch = TypedModelPatch::default();
                        for (ix, output) in res.into_iter().enumerate() {
                            let exotic_fact =
                                model.outlet_fact(OutletId::new(node.id, ix))?.exotic_fact.clone();

                            let name = if ix > 0 {
                                format!("{}.{ix}", node.name)
                            } else {
                                node.name.clone()
                            };
                            let wire = patch.wire_node(
                                name,
                                Const::new_with_opt_exotic_fact(
                                    output.into_arc_tensor(),
                                    exotic_fact,
                                )?,
                                &[],
                            )?[0];
                            patch.shunt_outside(model, (node.id, ix).into(), wire)?;
                        }
                        self.0 = node.id;
                        return Ok(Some(patch));
                    }
                    Err(e) => {
                        if !e.root_cause().is::<TooEarly>() {
                            Err(e).with_context(|| {
                                format!("Eager eval {node} during optimisation")
                            })?;
                        }
                    }
                }
            }
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::math;

    fn const_i64(model: &mut TypedModel, name: &str, shape: [usize; 2]) -> TractResult<OutletId> {
        let len = shape[0] * shape[1];
        let t = Tensor::from_shape(&shape, &(0..len as i64).collect::<Vec<_>>())?;
        model.add_const(name, t)
    }

    /// Sides are derived from the budget rather than fixed, so changing
    /// [`CONST_FOLD_MEM_BUDGET`] cannot silently move a case to the other side of
    /// the guard.
    fn side_for(bytes: u64) -> usize {
        (bytes / std::mem::size_of::<i64>() as u64).isqrt() as usize
    }

    /// Two consumers of one constant, both foldable. The constant is well over the
    /// element count the guard used to allow but inside the memory budget, so the
    /// graph must collapse entirely: once both consumers are constant the shared
    /// input is dead and nothing has been duplicated.
    #[test]
    fn fold_through_shared_const_inside_mem_budget() -> TractResult<()> {
        let mut model = TypedModel::default();
        let side = side_for(CONST_FOLD_MEM_BUDGET / 16);
        let shared = const_i64(&mut model, "shared", [side, side])?;
        let one = model.add_const("one", tensor2(&[[1i64]]))?;
        let two = model.add_const("two", tensor2(&[[2i64]]))?;
        let a = model.wire_node("a", math::mul(), &[shared, two])?[0];
        let b = model.wire_node("b", math::add(), &[shared, one])?[0];
        let sum = model.wire_node("sum", math::add(), &[a, b])?[0];
        model.select_output_outlets(&[sum])?;

        let decluttered = model.into_decluttered()?;
        let live: Vec<&str> = decluttered
            .nodes
            .iter()
            .filter(|n| !n.op_is::<Const>() && !n.op_is::<Dummy>())
            .map(|n| n.name.as_str())
            .collect();
        assert!(live.is_empty(), "expected a fully folded graph, got {live:?}");
        Ok(())
    }

    /// The same shape of graph with a constant over the budget keeps its consumers,
    /// so a large weight is not turned into one copy per consumer.
    #[test]
    fn keep_shared_const_over_mem_budget() -> TractResult<()> {
        let mut model = TypedModel::default();
        let side = side_for(CONST_FOLD_MEM_BUDGET * 2);
        let shared = const_i64(&mut model, "shared", [side, side])?;
        let one = model.add_const("one", tensor2(&[[1i64]]))?;
        let two = model.add_const("two", tensor2(&[[2i64]]))?;
        let a = model.wire_node("a", math::mul(), &[shared, two])?[0];
        let b = model.wire_node("b", math::add(), &[shared, one])?[0];
        model.select_output_outlets(&[a, b])?;

        let decluttered = model.into_decluttered()?;
        assert!(
            decluttered.nodes.iter().any(|n| !n.op_is::<Const>() && !n.op_is::<Dummy>()),
            "a shared constant over the memory budget must not be folded per consumer"
        );
        Ok(())
    }
}
