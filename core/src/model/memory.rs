use super::*;
use crate::prelude::*;
use std::collections::HashSet;
use std::fmt::Debug;
use std::fmt::Display;
use tract_data::internal::*;

/// Evaluate temporary memory usage with its related node at each step of the given order.
pub fn eval_tmp_memory_usage<F, O, Flushable>(
    model: &Graph<F, O>,
    order: &[usize],
    flushable: Flushable,
) -> TractResult<TVec<(usize, TDim)>>
where
    F: Fact + Clone + 'static,
    O: Debug + Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
    Flushable: Fn(&Node<F, O>) -> bool,
{
    let outputs = model.output_outlets()?.to_vec();

    let flush_lists = super::order::build_flush_list(model, order, &outputs, &flushable);
    let mut values: TVec<bool> = tvec![false; model.nodes.len()];

    let mut mem_by_steps: TVec<_> = tvec![(0, 0.into()); order.len()];

    let flushable_nodes = model
        .nodes()
        .iter()
        .filter(|node| (flushable)(node))
        .map(|it| it.id)
        .collect::<HashSet<_>>();

    for (step, n) in order.iter().enumerate() {
        let node = model.node(*n);

        for flush in flush_lists[step].iter() {
            values[*flush] = false;
        }

        // Active nodes are node that has not been flushed + inputs of the current node and current node.
        let mut step_active_nodes: HashSet<_> =
            values.iter().enumerate().filter_map(|(n, active)| active.then_some(n)).collect();

        step_active_nodes.extend(node.inputs.iter().map(|it| it.node));
        step_active_nodes.insert(*n);

        values[*n] = true;

        // Keep only flushable nodes.
        let step_active_flushable_nodes = step_active_nodes.intersection(&flushable_nodes);

        mem_by_steps[step] = (*n, 0.into());

        for n in step_active_flushable_nodes {
            let out_facts = model
                .node_output_facts(*n)?
                .into_iter()
                .map(|it| it.to_typed_fact())
                .collect::<TractResult<TVec<_>>>()?;
            mem_by_steps[step].1 += out_facts.iter().map(|it| it.mem_size()).sum::<TDim>();
        }
    }
    Ok(mem_by_steps)
}
