use std::fmt;
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ScopedNodeMemory {
    pub node: usize,
    pub scope: Scope,
    pub mem_size: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Scope {
    pub start: usize,
    pub end: usize,
}

impl Scope {
    fn is_disjoint(&self, other: &Scope) -> bool {
        self.start >= other.end || other.start >= self.end
    }

    pub fn is_alive_at_step(&self, step: usize) -> bool {
        self.start <= step && step < self.end
    }
}

pub fn eval_scoped_node_memories<F, O, Flushable>(
        model: &Graph<F, O>,
        order: &[usize],
        flushable: Flushable) -> TractResult<TVec<ScopedNodeMemory>>
where
    F: Fact + Clone + 'static,
    O: Debug + Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
    Flushable: Fn(&Node<F, O>) -> bool,
{

    let outputs = model.output_outlets()?.to_vec();
    let flush_lists = super::order::build_flush_list(model, order, &outputs, &flushable);
    let mut scoped_nodes = tvec![];

    for (step, n) in order.iter().enumerate() {
        let scope_start = step;
        let scope_end = flush_lists.iter().enumerate().find(|(_step, flush_list)| flush_list.contains(n))
                    .map(|it| usize::min(it.0 + 1, order.len()));

        let Some(scope_end) = scope_end else { continue; };

        let out_facts = model
                .node_output_facts(*n)?
                .into_iter()
                .map(|it| it.to_typed_fact())
                .collect::<TractResult<TVec<_>>>()?;


        scoped_nodes.push(ScopedNodeMemory {
            node: *n,
            scope: Scope {
                start: scope_start,
                end: scope_end
            },
            mem_size: out_facts.iter().map(|it| it.mem_size()).sum::<TDim>().to_usize()?,
        })
    }

    Ok(scoped_nodes)
}


#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MemoryPlan {
    pub by_partition: Vec<Vec<ScopedNodeMemory>>,
    pub by_steps: Vec<Vec<Option<ScopedNodeMemory>>>,
}

impl MemoryPlan {

    pub fn size_by_partition(&self) -> Vec<usize> {
        self.by_partition.iter().map(|it| it.iter().map(|it| it.mem_size).max().unwrap_or(0)).collect()
    }

    pub fn memory_size(&self) -> usize {
        self.by_partition.iter().map(|it| it.iter().map(|it| it.mem_size).max().unwrap_or(0)).sum()
    }
}

impl fmt::Display for MemoryPlan {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        for (step, mem_step) in self.by_steps.iter().enumerate() {
            writeln!(
               fmt,
               "step: {:5} => |{}|",
               step,
               mem_step.iter()
                       .map(|n| -> String { n.as_ref().map(|it| format!("{:^7}", it.node)).unwrap_or(format!("{:^7}", "*"))})
                       .collect::<Vec<String>>()
                       .join("|")
            )?;

        }
        writeln!(fmt, "memory_size: {}", self.memory_size())?;
        Ok(())
    }
}


pub fn eval_memory_plan<F, O, Flushable>(
        model: &Graph<F, O>,
        order: &[usize],
        flushable: Flushable) -> TractResult<MemoryPlan>
where
    F: Fact + Clone + 'static,
    O: Debug + Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
    Flushable: Fn(&Node<F, O>) -> bool,
{

    let mut scoped_node_memories = eval_scoped_node_memories(&model, &order, flushable)?;
    scoped_node_memories.sort_by_key(|it| it.mem_size);


    let mut partitions: Vec<Vec<ScopedNodeMemory>> = vec![];
    for node_mem in scoped_node_memories {
        // Find partitions where node scope is disjoint from existing.
        let mut available = partitions
                .iter_mut()
                .filter(|it| it.iter().all(|n| n.scope.is_disjoint(&node_mem.scope)))
                .collect::<Vec<_>>();

        // Find first available partition that has the max memory size
        available.sort_by_key(|n| n.iter().map(|it| it.mem_size as isize * -1).sum::<isize>());

        match available.first_mut() {
            Some(available) => {
                available.push(node_mem);
            },
            None => {
                partitions.push(vec![node_mem])
            },
        }
    }

    let by_steps: Vec<Vec<Option<ScopedNodeMemory>>> = (0..order.len())
        .map(|step| {
            let mem_step: Vec<_> = partitions.iter()
                .map(|p| {
                    p.iter().find(|it| it.scope.is_alive_at_step(step)).cloned()
                })
                .collect();
            ensure!(mem_step.len() <= partitions.len());
            Ok(mem_step)
        })
        .collect::<TractResult<Vec<_>>>()?;



    Ok(MemoryPlan { by_partition: partitions, by_steps })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::konst::Const;
    use crate::internal::*;
    use crate::ops::array::Gather;
    use crate::ops::math;

    #[test]
    fn test_node_scope() -> TractResult<()> {
        let mut model = TypedModel::default();
        let b = model.add_const("b", tensor1(&[0i64; 1000]))?; // 0
        let d = model.add_const("d", tensor1(&[0i64; 100]))?; // 1
        let a = model.add_source("a", i32::fact([10]))?; // 2
        let c = model.wire_node("c", Gather::new(0), &[a, b])?[0]; // 3
        let e = model.wire_node("e", Gather::new(0), &[c, d])?[0]; // 4
        model.set_output_outlets(&[e]).unwrap();

        let order = model.eval_order()?;
        let scoped_node_memory = eval_scoped_node_memories(&model, &order, |n| !n.op_is::<Const>())?;
        let plan = eval_memory_plan(&model, &order, |n| !n.op_is::<Const>())?;


        assert_eq!(order, &[2, 0, 3, 1, 4]);


        eprintln!("{model}");

        eprintln!("{:?}", order);
        eprintln!("{:#?}", scoped_node_memory);

        eprintln!("{plan}");

        // assert!(model.eval_order_opt_ram()?[2..] == [c.node, d.node, e.node]);
        Ok(())
    }
}


