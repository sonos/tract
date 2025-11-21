use std::fmt;
use std::fmt::Debug;
use tract_core::internal::num_integer::Integer;
use tract_core::internal::*;

use crate::fact::DeviceTypedFactExt;
use crate::sync::{DeviceSync, DeviceSyncKind};

/// Requirement for node outputs from a memory perspective.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NodeMemReq {
    pub outlet_id: OutletId,
    pub lifetime: Lifetime,
    pub mem_size: TDim,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Lifetime {
    pub start: usize,
    pub end: usize,
}

impl Lifetime {
    pub fn is_disjoint(&self, other: &Lifetime) -> bool {
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

fn next_nodes<'a>(model: &'a TypedModel, node: &TypedNode) -> Option<TVec<&'a TypedNode>> {
    if node.outputs.is_empty() {
        return None;
    };

    Some(
        node.outputs
            .iter()
            .flat_map(|o| {
                o.successors.iter().map(|succ| &model.nodes()[succ.node]).collect::<Vec<_>>()
            })
            .collect(),
    )
}

pub fn eval_device_mem_req_for_nodes(
    model: &TypedModel,
    order: &[usize],
) -> TractResult<TVec<NodeMemReq>> {
    let outputs = model.output_outlets()?.to_vec();
    let flush_lists = order::build_flush_list(model, order, &outputs, |node| {
        let Ok(facts) = model.node_output_facts(node.id) else { return false };

        let cpu_sync_in_next_nodes = next_nodes(model, node).is_some_and(|nodes| {
            nodes.iter().any(|it| {
                it.op_as::<DeviceSync>().is_some_and(|op| op.kind == DeviceSyncKind::ToHost)
            })
        });

        !cpu_sync_in_next_nodes
            && facts.iter().any(|it| {
                it.as_device_fact()
                    .map(|it| it.is_from_device() && !it.is_state_owned())
                    .unwrap_or(false)
            })
    });
    let mut scoped_nodes = tvec![];

    for (step, n) in order.iter().enumerate() {
        let lifetime_start = step;

        let lifetime_end = flush_lists
            .iter()
            .enumerate()
            .find(|(_step, flush_list)| flush_list.contains(n))
            .map(|it| usize::min(it.0 + 1, order.len()));
        // Ignore nodes that won't be flushed from Device.
        let Some(lifetime_end) = lifetime_end else {
            continue;
        };

        let out_device_tmp_facts = model
            .node_output_facts(*n)?
            .into_iter()
            .flat_map(|it| it.as_device_fact())
            .filter(|it| it.is_from_device())
            .collect::<TVec<_>>();

        if out_device_tmp_facts.is_empty() {
            continue;
        }

        for (slot, fact) in out_device_tmp_facts.iter().enumerate() {
            let outlet_id = OutletId { node: *n, slot };
            for buff_size in fact.buffer_sizes() {
                scoped_nodes.push(NodeMemReq {
                    outlet_id,
                    lifetime: Lifetime { start: lifetime_start, end: lifetime_end },
                    mem_size: buff_size,
                })
            }
        }
    }

    Ok(scoped_nodes)
}

fn collect_opaque_facts(model: &TypedModel) -> TractResult<Vec<NodeOpaqueFacts>> {
    let mut res: Vec<TVec<Option<Box<dyn OpaqueFact>>>> = vec![];
    for node in model.nodes() {
        let mut tmp: TVec<Option<Box<dyn OpaqueFact>>> = tvec![];
        for fact in model.node_output_facts(node.id)? {
            if let Some(dev_fact) = fact.as_device_fact() {
                tmp.push(dev_fact.opaque_fact.clone());
            }
        }
        res.push(tmp);
    }
    Ok(res)
}

/// A partition is a list of node that have disjoint memory requirement from a lifetime
/// perspective.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Partition {
    pub nodes: Vec<NodeMemReq>,
}

impl Partition {
    pub fn eval_size_to_i64(&self, symbols: &SymbolValues) -> TractResult<i64> {
        let mut max_size = self
            .nodes
            .iter()
            .map(|it| it.mem_size.eval_to_i64(symbols))
            .collect::<TractResult<Vec<_>>>()?
            .into_iter()
            .max()
            .unwrap_or(0);
        max_size = Integer::next_multiple_of(&max_size, &(vector_size() as i64));
        Ok(max_size)
    }

    pub fn size(&self) -> TDim {
        TDim::Max(self.nodes.iter().map(|s| s.mem_size.clone()).collect()).simplify()
    }

    pub fn has_no_conflict_with_lifetime(&self, lifetime: &Lifetime) -> bool {
        self.nodes.iter().all(|n| n.lifetime.is_disjoint(lifetime))
    }

    pub fn find_node_alive_at_step(&self, step: usize) -> Option<&NodeMemReq> {
        self.nodes.iter().find(|it| it.lifetime.is_alive_at_step(step))
    }
}

type NodeOpaqueFacts = TVec<Option<Box<dyn OpaqueFact>>>;
/// This struct represents a resolved memory schema for a model that contains
/// GPU operators. This schema is concrete.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceResolvedMemSchema {
    pub offsets_by_node: Vec<Option<TVec<TVec<usize>>>>,
    pub memory_size: usize,
}

/// This struct represent a memory schema for node output memory that are handled
/// by a GPU.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DeviceMemSchema {
    /// Total numbef in the model.
    pub model_num_nodes: usize,
    pub by_partition: Vec<Partition>,
    // vec![vec![Option<NodeMemReq>; num_partitions]; num_steps].
    pub by_steps: Vec<Vec<Option<NodeMemReq>>>,
    pub opaque_facts: Vec<NodeOpaqueFacts>,
}

impl DeviceMemSchema {
    /// Returns memory size of each inner partitions.
    pub fn size_by_partition(&self) -> Vec<TDim> {
        self.by_partition.iter().map(|it| it.size()).collect()
    }

    /// Evaluate memory size by partition for given symbol values.
    pub fn eval_size_by_partition(&self, symbols: &SymbolValues) -> TractResult<Vec<i64>> {
        self.by_partition.iter().map(|it| it.eval_size_to_i64(symbols)).collect()
    }

    /// Returns total memory size required for the schema.
    pub fn memory_size(&self) -> TDim {
        self.by_partition.iter().map(|it| it.size()).sum()
    }

    /// Evaluate memory size required for the schema for given symbol values.
    pub fn eval_memory_size(&self, symbols: &SymbolValues) -> TractResult<i64> {
        self.by_partition.iter().map(|it| it.eval_size_to_i64(symbols)).sum()
    }

    /// Compute offsets for each node for given symbols. Node ids
    /// are indexes in the returned vector.
    pub fn compute_offset_by_node(
        &self,
        symbols: &SymbolValues,
    ) -> TractResult<Vec<Option<TVec<TVec<usize>>>>> {
        let mut cursor = 0;
        let mut offset_by_outlet: Vec<Option<TVec<TVec<usize>>>> = vec![None; self.model_num_nodes];

        for partition in &self.by_partition {
            for node_mem in &partition.nodes {
                let node = node_mem.outlet_id.node;
                let slot = node_mem.outlet_id.slot;

                let slots: &mut TVec<TVec<usize>> =
                    offset_by_outlet[node].get_or_insert_with(|| tvec![tvec!()]);

                if slot < 1 {
                    slots[slot].push(cursor);
                } else {
                    if slots.len() <= slot {
                        slots.resize_with(slot + 1, TVec::<usize>::new);
                    }
                    slots[slot].push(cursor);
                }
            }
            cursor += partition.eval_size_to_i64(symbols)? as usize;
        }

        Ok(offset_by_outlet)
    }

    /// Evaluate peak memory size for given symbols. The return value is lower or equal to the memory
    /// size of the schema. The difference between peak memory size and memory size represents the
    /// memory fragmentation introduced by the schema.
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

    /// Evaluate the usage for given symbols as the ratio between
    /// schema memory size and peak memory size. A value of 1.0 means
    /// that the schema doesn't introduce memory fragmentation.
    pub fn eval_usage(&self, symbols: &SymbolValues) -> TractResult<f32> {
        let memory_size = self.eval_memory_size(symbols)? as f32;
        let peak_memory_size = self.eval_peak_memory_size(symbols)? as f32;
        Ok(peak_memory_size / memory_size)
    }
}

impl fmt::Display for DeviceMemSchema {
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
                            .map(|it| format!("{:^7}/{:^7}", it.outlet_id.node, it.outlet_id.slot))
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

impl DeviceMemSchema {
    /// Resolve Memory schema with given symbols.
    pub fn resolve(&self, symbols: &SymbolValues) -> TractResult<DeviceResolvedMemSchema> {
        Ok(DeviceResolvedMemSchema {
            offsets_by_node: self.compute_offset_by_node(symbols)?,
            memory_size: self.eval_memory_size(symbols)?.try_into()?,
        })
    }

    /// Build a memory schema for given model and execution order. The hint is used to optimize
    /// the memory schema because it is based on symbolic dimensions. That doesn't mean it will be
    /// optimal for all possible values for symbolic dimensions.
    pub fn build(
        model: &TypedModel,
        order: &[usize],
        hint: &SymbolValues,
    ) -> TractResult<DeviceMemSchema> {
        let mut nodes_mem_req = eval_device_mem_req_for_nodes(model, order)?;

        let opaque_facts = collect_opaque_facts(model)?;
        let hinted_mem_size = nodes_mem_req
            .iter()
            .map(|node_mem| Ok((node_mem.outlet_id, node_mem.mem_size.eval_to_i64(hint)?)))
            .collect::<TractResult<HashMap<OutletId, i64>>>()?;

        nodes_mem_req.sort_by(|lhs, rhs| {
            let lhs_hint_mem_size = hinted_mem_size.get(&lhs.outlet_id);
            let rhs_hint_mem_size = hinted_mem_size.get(&rhs.outlet_id);
            lhs_hint_mem_size.cmp(&rhs_hint_mem_size).reverse()
        });

        let mut partitions: Vec<Partition> = vec![];
        for node_mem in nodes_mem_req {
            // Find partitions where node lifetime is disjoint from existing.
            let mut available = partitions
                .iter_mut()
                .filter(|it| it.has_no_conflict_with_lifetime(&node_mem.lifetime))
                .collect::<Vec<_>>();

            available.sort_by_cached_key(|n| {
                -n.nodes.iter().flat_map(|it| hinted_mem_size.get(&it.outlet_id)).sum::<i64>()
            });

            match available.first_mut() {
                Some(available) => {
                    available.nodes.push(node_mem);
                }
                None => partitions.push(Partition { nodes: vec![node_mem] }),
            }
        }

        let by_steps: Vec<Vec<Option<NodeMemReq>>> = (0..order.len())
            .map(|step| {
                let mem_step: Vec<_> =
                    partitions.iter().map(|p| p.find_node_alive_at_step(step).cloned()).collect();
                ensure!(mem_step.len() <= partitions.len());
                Ok(mem_step)
            })
            .collect::<TractResult<Vec<_>>>()?;

        Ok(DeviceMemSchema {
            model_num_nodes: model.nodes().len(),
            by_partition: partitions,
            by_steps,
            opaque_facts,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lifetime_is_disjoint() {
        let l1 = Lifetime { start: 0, end: 5 };
        let l2 = Lifetime { start: 5, end: 10 };
        let l3 = Lifetime { start: 3, end: 7 };

        assert!(l1.is_disjoint(&l2));
        assert!(l2.is_disjoint(&l1));
        assert!(!l1.is_disjoint(&l3));
        assert!(!l3.is_disjoint(&l2));
    }

    #[test]
    fn test_lifetime_is_alive_at_step() {
        let lifetime = Lifetime { start: 5, end: 10 };

        assert!(!lifetime.is_alive_at_step(4));
        assert!(lifetime.is_alive_at_step(5));
        assert!(lifetime.is_alive_at_step(7));
        assert!(lifetime.is_alive_at_step(9));
        assert!(!lifetime.is_alive_at_step(10));
    }

    #[test]
    fn test_empty_lifetime() {
        let lifetime = Lifetime { start: 5, end: 5 };
        assert!(lifetime.is_empty());
        assert_eq!(lifetime.len(), 0);
    }

    #[test]
    fn test_node_mem_req_basic() {
        let outlet_id = OutletId { node: 1, slot: 0 };
        let req = NodeMemReq {
            outlet_id,
            lifetime: Lifetime { start: 0, end: 5 },
            mem_size: 1000.into(),
        };

        assert_eq!(req.outlet_id.node, 1);
        assert_eq!(req.lifetime.start, 0);
        assert_eq!(req.lifetime.end, 5);
        assert_eq!(req.mem_size.to_i64().unwrap(), 1000);
    }

    #[test]
    fn test_partition_has_no_conflict() {
        let outlet_id = OutletId { node: 1, slot: 0 };
        let node1 = NodeMemReq {
            outlet_id,
            lifetime: Lifetime { start: 0, end: 5 },
            mem_size: 1000.into(),
        };

        let partition = Partition { nodes: vec![node1] };

        assert!(partition.has_no_conflict_with_lifetime(&Lifetime { start: 5, end: 10 }));
        assert!(!partition.has_no_conflict_with_lifetime(&Lifetime { start: 3, end: 7 }));
    }

    #[test]
    fn test_partition_find_node() {
        let outlet_id = OutletId { node: 1, slot: 0 };
        let node1 = NodeMemReq {
            outlet_id,
            lifetime: Lifetime { start: 0, end: 5 },
            mem_size: 1000.into(),
        };

        let outlet_id = OutletId { node: 2, slot: 0 };
        let node2 = NodeMemReq {
            outlet_id,
            lifetime: Lifetime { start: 5, end: 10 },
            mem_size: 2000.into(),
        };

        let partition = Partition { nodes: vec![node1.clone(), node2.clone()] };

        assert_eq!(partition.find_node_alive_at_step(3), Some(&node1));
        assert_eq!(partition.find_node_alive_at_step(7), Some(&node2));
        assert_eq!(partition.find_node_alive_at_step(10), None);
    }
}
