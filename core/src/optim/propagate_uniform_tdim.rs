use crate::internal::*;
use crate::optim::OptimizerSession;

/// Forward pass that refreshes `TypedFact::uniform_tdim` annotations by
/// re-running each op's `output_facts` against its current input facts.
///
/// The default declutter pipeline computes a node's `uniform_tdim` once at
/// load time, then reuses the cached fact.  Some declutter rewrites
/// (notably `Iff` folding when the condition is provably constant) shunt a
/// node's input edge without re-running the consumer's `output_facts` —
/// the consumer's cached fact then references the stale upstream fact and
/// loses any newly-available `uniform_tdim` annotation.  Every wire
/// downstream of the shunt then sees `uniform_tdim = None`, and passes
/// like `FoldUniformMask` or Blockify section detection silently miss it.
///
/// This pass walks the model in topological order, calls `output_facts`
/// fresh on each node, and copies the recomputed `uniform_tdim` over the
/// cached one when it differs.  Other fact fields are untouched (the
/// existing declutter loop is responsible for them).  Iterates to fixpoint
/// since a refreshed annotation upstream may unlock more refreshes
/// downstream.
#[derive(Clone, Debug, Default)]
pub struct PropagateUniformTdim;

impl super::TypedPass for PropagateUniformTdim {
    fn reset(&mut self) -> TractResult<()> {
        Ok(())
    }

    fn next(
        &mut self,
        _session: &mut OptimizerSession,
        _model: &TypedModel,
    ) -> TractResult<Option<TypedModelPatch>> {
        Ok(None)
    }

    fn run_direct(&mut self, model: &mut TypedModel) -> TractResult<bool> {
        let order = model.eval_order()?;
        let mut any_changed = false;
        loop {
            let mut changed = false;
            for &node_id in &order {
                let typed_op = match model.nodes()[node_id].op.as_typed() {
                    Some(op) => op,
                    None => continue,
                };
                let input_facts: TVec<TypedFact> = model.nodes()[node_id]
                    .inputs
                    .iter()
                    .map(|i| model.outlet_fact(*i).cloned())
                    .collect::<TractResult<_>>()?;
                let input_refs: TVec<&TypedFact> = input_facts.iter().collect();
                let new_facts = match typed_op.output_facts(&input_refs) {
                    Ok(f) => f,
                    Err(_) => continue,
                };
                for (slot, new_fact) in new_facts.iter().enumerate() {
                    let current_uniform_tdim =
                        model.nodes()[node_id].outputs[slot].fact.uniform_tdim.clone();
                    if current_uniform_tdim != new_fact.uniform_tdim {
                        model.nodes_mut()[node_id].outputs[slot].fact.uniform_tdim =
                            new_fact.uniform_tdim.clone();
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
            any_changed = true;
        }
        Ok(any_changed)
    }
}
