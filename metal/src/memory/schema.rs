use crate::fact::MetalTypedFactExt;
use std::fmt;
use std::fmt::Debug;
use tract_core::internal::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ScopedNodeMemory {
    pub node: usize,
    pub scope: Scope,
    pub mem_size: TDim,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Scope {
    pub start: usize,
    pub end: usize,
}

impl Scope {
    pub fn is_disjoint(&self, other: &Scope) -> bool {
        self.start >= other.end || other.start >= self.end
    }

    pub fn is_alive_at_step(&self, step: usize) -> bool {
        self.start <= step && step < self.end
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn len(&self) -> usize {
        self.end - self.start
    }
}

pub fn eval_metal_scope_node_mem(
    model: &TypedModel,
    order: &[usize],
) -> TractResult<TVec<ScopedNodeMemory>> {
    let outputs = model.output_outlets()?.to_vec();
    let flush_lists = order::build_flush_list(model, order, &outputs, |node| {
        let Ok(facts) = model.node_output_facts(node.id) else { return false };

        facts.iter().any(|it| it.to_metal_fact().map(|it| it.is_from_gpu()).unwrap_or(false))
    });
    let mut scoped_nodes = tvec![];

    for (step, n) in order.iter().enumerate() {
        let scope_start = step;
        let scope_end = flush_lists
            .iter()
            .enumerate()
            .find(|(_step, flush_list)| flush_list.contains(n))
            .map(|it| usize::min(it.0 + 1, order.len()));

        let Some(scope_end) = scope_end else {
            continue;
        };

        let out_metal_tmp_facts = model
            .node_output_facts(*n)?
            .into_iter()
            .flat_map(|it| it.to_metal_fact().ok())
            .filter(|it| it.is_from_gpu())
            .collect::<TVec<_>>();

        if out_metal_tmp_facts.is_empty() {
            continue;
        }

        scoped_nodes.push(ScopedNodeMemory {
            node: *n,
            scope: Scope { start: scope_start, end: scope_end },
            mem_size: out_metal_tmp_facts.iter().map(|it| it.mem_size()).sum::<TDim>(),
        })
    }

    Ok(scoped_nodes)
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Partition {
    pub nodes: Vec<ScopedNodeMemory>,
}

impl Partition {
    pub fn eval_size_to_i64(&self, symbols: &SymbolValues) -> TractResult<i64> {
        Ok(self
            .nodes
            .iter()
            .map(|it| it.mem_size.eval_to_i64(symbols))
            .collect::<TractResult<Vec<_>>>()?
            .into_iter()
            .max()
            .unwrap_or(0))
    }

    pub fn size(&self) -> TDim {
        TDim::Max(self.nodes.iter().map(|s| s.mem_size.clone()).collect())
    }

    pub fn is_disjoint(&self, scope: &Scope) -> bool {
        self.nodes.iter().all(|n| n.scope.is_disjoint(scope))
    }

    pub fn find_node_alive_at_step(&self, step: usize) -> Option<&ScopedNodeMemory> {
        self.nodes.iter().find(|it| it.scope.is_alive_at_step(step))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalResolvedMemSchema {
    pub offsets_by_node: Vec<usize>,
    pub memory_size: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MetalMemSchema {
    pub by_partition: Vec<Partition>,
    pub by_steps: Vec<Vec<Option<ScopedNodeMemory>>>,
}

impl MetalMemSchema {
    pub fn eval_size_by_partition(&self, symbols: &SymbolValues) -> TractResult<Vec<i64>> {
        self.by_partition.iter().map(|it| it.eval_size_to_i64(symbols)).collect()
    }

    pub fn size_by_partition(&self) -> Vec<TDim> {
        self.by_partition.iter().map(|it| it.size()).collect()
    }

    pub fn memory_size(&self) -> TDim {
        self.by_partition.iter().map(|it| it.size()).sum()
    }

    pub fn eval_memory_size(&self, symbols: &SymbolValues) -> TractResult<i64> {
        self.by_partition.iter().map(|it| it.eval_size_to_i64(symbols)).sum()
    }

    pub fn compute_offset_by_node(
        &self,
        num_nodes: usize,
        symbols: &SymbolValues,
    ) -> TractResult<Vec<usize>> {
        let mut cursor = 0;
        let mut offset_by_node = vec![0; num_nodes];

        for partition in self.by_partition.iter() {
            for node_mem in partition.nodes.iter() {
                offset_by_node[node_mem.node] = cursor;
            }
            cursor += partition.eval_size_to_i64(symbols)? as usize;
        }

        Ok(offset_by_node)
    }

    pub fn eval_peak_memory_size(&self, symbols: &SymbolValues) -> TractResult<i64> {
        Ok(self
            .by_steps
            .iter()
            .map(|active_nodes| {
                active_nodes
                    .iter()
                    .flatten()
                    .map(|it| it.mem_size.clone())
                    .sum::<TDim>()
                    .eval_to_i64(symbols)
            })
            .collect::<TractResult<Vec<_>>>()?
            .into_iter()
            .max()
            .unwrap_or(0))
    }

    pub fn eval_usage(&self, symbols: &SymbolValues) -> TractResult<f32> {
        let memory_size = self.eval_memory_size(symbols)? as f32;
        let peak_memory_size = self.eval_peak_memory_size(symbols)? as f32;
        Ok(peak_memory_size / memory_size)
    }
}

impl fmt::Display for MetalMemSchema {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        for (step, mem_step) in self.by_steps.iter().enumerate() {
            writeln!(
                fmt,
                "step: {:5} => |{}|",
                step,
                mem_step
                    .iter()
                    .map(|n| -> String {
                        n.as_ref()
                            .map(|it| format!("{:^7}", it.node))
                            .unwrap_or(format!("{:^7}", "*"))
                    })
                    .collect::<Vec<String>>()
                    .join("|")
            )?;
        }
        writeln!(fmt, "memory_size: {}", self.memory_size())?;
        Ok(())
    }
}

impl MetalMemSchema {
    pub fn resolve(
        &self,
        num_nodes: usize,
        symbols: &SymbolValues,
    ) -> TractResult<MetalResolvedMemSchema> {
        Ok(MetalResolvedMemSchema {
            offsets_by_node: self.compute_offset_by_node(num_nodes, symbols)?,
            memory_size: self.eval_memory_size(symbols)?.try_into()?,
        })
    }

    pub fn build(
        model: &TypedModel,
        order: &[usize],
        hint: &SymbolValues,
    ) -> TractResult<MetalMemSchema> {
        let mut scoped_nodes_mem = eval_metal_scope_node_mem(model, order)?;

        let hinted_mem_size = scoped_nodes_mem
            .iter()
            .map(|node_mem| Ok((node_mem.node, node_mem.mem_size.eval_to_i64(hint)?)))
            .collect::<TractResult<HashMap<usize, i64>>>()?;

        scoped_nodes_mem.sort_by(|lhs, rhs| {
            let lhs_hint_mem_size = hinted_mem_size.get(&lhs.node);
            let rhs_hint_mem_size = hinted_mem_size.get(&rhs.node);

            lhs.scope
                .end
                .cmp(&rhs.scope.end)
                .reverse()
                .then(lhs.scope.len().cmp(&rhs.scope.len()).reverse())
                .then(lhs_hint_mem_size.cmp(&rhs_hint_mem_size).reverse())
        });

        let mut partitions: Vec<Partition> = vec![];
        for node_mem in scoped_nodes_mem {
            // Find partitions where node scope is disjoint from existing.
            let mut available = partitions
                .iter_mut()
                .filter(|it| it.is_disjoint(&node_mem.scope))
                .collect::<Vec<_>>();

            available.sort_by_cached_key(|n| {
                -n.nodes.iter().flat_map(|it| hinted_mem_size.get(&it.node)).sum::<i64>()
            });

            match available.first_mut() {
                Some(available) => {
                    available.nodes.push(node_mem);
                }
                None => partitions.push(Partition { nodes: vec![node_mem] }),
            }
        }

        let by_steps: Vec<Vec<Option<ScopedNodeMemory>>> = (0..order.len())
            .map(|step| {
                let mem_step: Vec<_> =
                    partitions.iter().map(|p| p.find_node_alive_at_step(step).cloned()).collect();
                ensure!(mem_step.len() <= partitions.len());
                Ok(mem_step)
            })
            .collect::<TractResult<Vec<_>>>()?;

        Ok(MetalMemSchema { by_partition: partitions, by_steps })
    }
}
