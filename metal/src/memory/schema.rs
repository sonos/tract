use crate::fact::MetalTypedFactExt;
use crate::ops::MetalSyncKind;
use std::fmt;
use std::fmt::Debug;
use tract_core::internal::*;

/// Requirement for node outputs from a memory perspective.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NodeMemReq {
    pub node: usize,
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

pub fn eval_metal_mem_req_for_nodes(
    model: &TypedModel,
    order: &[usize],
) -> TractResult<TVec<NodeMemReq>> {
    let outputs = model.output_outlets()?.to_vec();
    let flush_lists = order::build_flush_list(model, order, &outputs, |node| {
        let Ok(facts) = model.node_output_facts(node.id) else { return false };

        let cpu_sync_in_next_nodes = next_nodes(model, node).is_some_and(|nodes| {
            nodes.iter().any(|it| {
                it.op_as::<crate::ops::MetalSync>()
                    .is_some_and(|op| op.kind == MetalSyncKind::ToCpu)
            })
        });

        !cpu_sync_in_next_nodes
            && facts.iter().any(|it| it.to_metal_fact().map(|it| it.is_from_gpu()).unwrap_or(false))
    });
    let mut scoped_nodes = tvec![];

    for (step, n) in order.iter().enumerate() {
        let lifetime_start = step;
        let lifetime_end = flush_lists
            .iter()
            .enumerate()
            .find(|(_step, flush_list)| flush_list.contains(n))
            .map(|it| usize::min(it.0 + 1, order.len()));

        // Ignore nodes that won't be flushed from gpu.
        let Some(lifetime_end) = lifetime_end else {
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

        scoped_nodes.push(NodeMemReq {
            node: *n,
            lifetime: Lifetime { start: lifetime_start, end: lifetime_end },
            mem_size: out_metal_tmp_facts.iter().map(|it| it.mem_size()).sum::<TDim>(),
        })
    }

    Ok(scoped_nodes)
}

/// A partition is a list of node that have disjoint memory requirement from a lifetime
/// perspective.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Partition {
    pub nodes: Vec<NodeMemReq>,
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

    pub fn has_no_conflict_with_lifetime(&self, lifetime: &Lifetime) -> bool {
        self.nodes.iter().all(|n| n.lifetime.is_disjoint(lifetime))
    }

    pub fn find_node_alive_at_step(&self, step: usize) -> Option<&NodeMemReq> {
        self.nodes.iter().find(|it| it.lifetime.is_alive_at_step(step))
    }
}

/// This struct represents a resolved memory schema for a model that contains
/// Metal operators. This schema is concrete.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalResolvedMemSchema {
    pub offsets_by_node: Vec<Option<usize>>,
    pub memory_size: usize,
}

/// This struct represent a memory schema for node output memory that are handled
/// by Metal GPU.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MetalMemSchema {
    /// Total numbef in the model.
    pub model_num_nodes: usize,
    pub by_partition: Vec<Partition>,
    // vec![vec![Option<NodeMemReq>; num_partitions]; num_steps].
    pub by_steps: Vec<Vec<Option<NodeMemReq>>>,
}

impl MetalMemSchema {
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
    ) -> TractResult<Vec<Option<usize>>> {
        let mut cursor = 0;
        let mut offset_by_node = vec![None; self.model_num_nodes];

        for partition in self.by_partition.iter() {
            for node_mem in partition.nodes.iter() {
                offset_by_node[node_mem.node] = Some(cursor);
            }
            cursor += partition.eval_size_to_i64(symbols)? as usize;
        }

        Ok(offset_by_node)
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
    /// Resolve Memory schema with given symbols.
    pub fn resolve(&self, symbols: &SymbolValues) -> TractResult<MetalResolvedMemSchema> {
        Ok(MetalResolvedMemSchema {
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
    ) -> TractResult<MetalMemSchema> {
        let mut nodes_mem_req = eval_metal_mem_req_for_nodes(model, order)?;

        let hinted_mem_size = nodes_mem_req
            .iter()
            .map(|node_mem| Ok((node_mem.node, node_mem.mem_size.eval_to_i64(hint)?)))
            .collect::<TractResult<HashMap<usize, i64>>>()?;

        nodes_mem_req.sort_by(|lhs, rhs| {
            let lhs_hint_mem_size = hinted_mem_size.get(&lhs.node);
            let rhs_hint_mem_size = hinted_mem_size.get(&rhs.node);
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
                -n.nodes.iter().flat_map(|it| hinted_mem_size.get(&it.node)).sum::<i64>()
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

        Ok(MetalMemSchema {
            model_num_nodes: model.nodes().len(),
            by_partition: partitions,
            by_steps,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MetalTransform;
    use tract_core::ops::einsum::BasicMatMul;
    use tract_core::ops::math::{add, mul};
    use tract_core::ops::nn::{Softmax, SoftmaxExp};
    use tract_core::transform::ModelTransform;

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
        let req =
            NodeMemReq { node: 1, lifetime: Lifetime { start: 0, end: 5 }, mem_size: 1000.into() };

        assert_eq!(req.node, 1);
        assert_eq!(req.lifetime.start, 0);
        assert_eq!(req.lifetime.end, 5);
        assert_eq!(req.mem_size.to_i64().unwrap(), 1000);
    }

    #[test]
    fn test_partition_has_no_conflict() {
        let node1 =
            NodeMemReq { node: 1, lifetime: Lifetime { start: 0, end: 5 }, mem_size: 1000.into() };

        let partition = Partition { nodes: vec![node1] };

        assert!(partition.has_no_conflict_with_lifetime(&Lifetime { start: 5, end: 10 }));
        assert!(!partition.has_no_conflict_with_lifetime(&Lifetime { start: 3, end: 7 }));
    }

    #[test]
    fn test_partition_find_node() {
        let node1 =
            NodeMemReq { node: 1, lifetime: Lifetime { start: 0, end: 5 }, mem_size: 1000.into() };

        let node2 =
            NodeMemReq { node: 2, lifetime: Lifetime { start: 5, end: 10 }, mem_size: 2000.into() };

        let partition = Partition { nodes: vec![node1.clone(), node2.clone()] };

        assert_eq!(partition.find_node_alive_at_step(3), Some(&node1));
        assert_eq!(partition.find_node_alive_at_step(7), Some(&node2));
        assert_eq!(partition.find_node_alive_at_step(10), None);
    }

    fn wire_sdpa_layer(
        model: &mut TypedModel,
        name: impl ToString,
        q: OutletId,
        k: OutletId,
        v: OutletId,
    ) -> TractResult<TVec<OutletId>> {
        let name = name.to_string();

        // Reshape Q
        let q_shape = model.outlet_fact(q)?.shape.to_tvec();
        let embed_dim: TDim = q_shape[1].clone();
        let head_dim: TDim = q_shape[3].clone();
        let batch: TDim = q_shape[0].clone();
        let seq_len: TDim = q_shape[2].clone();
        ensure!(batch.to_i64()? == 1, "Input 'q' shape is {:?} (expect batch = 1)", q_shape);
        ensure!(q_shape.len() == 4, "Input 'q' shape is {:?} (expect 4D)", q_shape);
        let q_reshaped = model.wire_node(
            format!("q_reshape_{}", name),
            AxisOp::Reshape(
                0,
                q_shape.clone(),
                tvec![embed_dim.clone(), batch.clone(), seq_len.clone(), head_dim.clone(),],
            ),
            &[q],
        )?[0];

        // Reshape K
        let k_shape = model.outlet_fact(k)?.shape.to_tvec();
        ensure!(k_shape.len() == 4, "Input 'k' shape is {:?} (expect 4D)", k_shape);
        let seq_plus_prompt_len: TDim = k_shape[2].clone();

        let k_reshaped = model.wire_node(
            format!("k_reshape_{}", name),
            AxisOp::Reshape(
                0,
                k_shape.clone(),
                tvec![
                    embed_dim.clone(),
                    batch.clone(),
                    seq_plus_prompt_len.clone(),
                    head_dim.clone(),
                ],
            ),
            &[k],
        )?[0];

        // Compute Q * K^T
        let qk = model.wire_node(
            format!("qk_{}", name),
            BasicMatMul {
                transpose_a: false,
                transpose_b: true,
                transpose_c: false,
                quantize_output: None,
            },
            &[q_reshaped, k_reshaped],
        )?[0];

        let qk_squeezed = model.wire_node(
            format!("qk_squeezed_{}", name),
            AxisOp::Reshape(
                0,
                tvec![
                    embed_dim.clone(),
                    batch.clone(),
                    seq_len.clone(),
                    seq_plus_prompt_len.clone(),
                ],
                tvec![embed_dim.clone(), seq_len.clone(), seq_plus_prompt_len.clone(),],
            ),
            &[qk],
        )?[0];

        // Scale factor for attention
        let scale = model.add_const(
            format!("scale_{}", name),
            tensor3(&[[[1.0f32 / (head_dim.to_i64()? as f32).sqrt()]]]),
        )?;
        let qk_scaled =
            model.wire_node(format!("qk_scaled_{}", name), mul(), &[qk_squeezed, scale])?[0];

        // Mask QK
        let mask = model.add_const("mask", tensor3(&[[[1.0f32]]]))?;
        let qk_scaled_masked =
            model.wire_node(format!("qk_scaled_masked_{}", name), add(), &[qk_scaled, mask])?[0];

        // Apply softmax
        let attention = model.wire_node(
            format!("attention_weights_{}", name),
            Softmax::new(tvec![2], None, SoftmaxExp::Libc),
            &[qk_scaled_masked],
        )?[0];

        // Reshape V
        let v_reshaped = model.wire_node(
            format!("v_reshape_{}", name),
            AxisOp::Reshape(
                0,
                k_shape,
                tvec![embed_dim.clone(), seq_plus_prompt_len.clone(), head_dim.clone(),],
            ),
            &[v],
        )?[0];

        // Multiply with V
        let output = model.wire_node(
            format!("attention_output_{}", name),
            BasicMatMul {
                transpose_a: false,
                transpose_b: false,
                transpose_c: false,
                quantize_output: None,
            },
            &[attention, v_reshaped],
        )?[0];

        // Reshape output
        let output_reshaped = model.wire_node(
            format!("output_reshape_{}", name),
            AxisOp::Reshape(
                0,
                tvec![embed_dim.clone(), seq_len.clone(), head_dim.clone(),],
                q_shape,
            ),
            &[output],
        )?;
        Ok(output_reshaped)
    }

    #[test]
    fn test_build_schema_from_model() -> TractResult<()> {
        // Given
        const EMBED_DIM: i64 = 32;
        const HEAD_DIM: i64 = 64;
        const SEQUENCE_LENGTH: i64 = 1;
        const PAST_SEQUENCE_LENGTH: i64 = 8;
        const EXPECTED_PEAK_SIZE: i64 = 9344;
        const EXPECTED_USAGE: f32 = 0.89;

        // Build a model with Scaled Dot-Product Attention (SDPA) layers
        let mut model = TypedModel::default();

        // Input shapes for Q, K, V
        let s = TDim::Sym(model.sym("S"));
        let p = TDim::Sym(model.sym("P"));
        let q_fact = f32::fact(tvec![1.into(), EMBED_DIM.into(), s.clone(), HEAD_DIM.into()]);
        let k_fact = f32::fact(tvec![1.into(), EMBED_DIM.into(), s + p, HEAD_DIM.into()]);
        let v_fact = k_fact.clone();

        // Create inputs for Q, K, V
        let q = model.add_source("q", q_fact)?;
        let k = model.add_source("k", k_fact)?;
        let v = model.add_source("v", v_fact)?;

        let outputs = wire_sdpa_layer(&mut model, "0", q, k, v)?;
        let outputs = wire_sdpa_layer(&mut model, "1", outputs[0], k, v)?;

        model.set_output_outlets(&outputs)?;

        // Transform model for Metal execution
        let model = MetalTransform::default().transform_into(model)?;

        // Get execution order
        let order = model.eval_order()?;

        // Hint symbol values
        let mut symbol_values = SymbolValues::default();
        symbol_values.set(&model.symbols.get("S").context("Missing symbol S")?, SEQUENCE_LENGTH);
        symbol_values
            .set(&model.symbols.get("P").context("Missing symbol P")?, PAST_SEQUENCE_LENGTH);

        // Build memory schema
        let schema = MetalMemSchema::build(&model, &order, &symbol_values)?;

        // Verify number of nodes
        assert!(schema.model_num_nodes > 1, "Schema should contain at least 2 nodes");

        // Verify number of partitions
        assert!(schema.by_partition.len() > 1, "Schema should contain at least 2 partitions");

        // Verify steps
        assert_eq!(schema.by_steps.len(), order.len());
        for step in 0..schema.by_steps.len() {
            for partition in schema.by_partition.iter() {
                let partition_size = partition.eval_size_to_i64(&symbol_values)?;

                // No empty partition
                assert!(!partition.nodes.is_empty());

                if let Some(this) = partition.find_node_alive_at_step(step) {
                    // Node memory requirement should be <= the partition size
                    let node_size = this.mem_size.eval_to_i64(&symbol_values)?;
                    assert!(node_size <= partition_size);
                    assert!(node_size > 0);

                    // All nodes should have a valid lifetime
                    assert!(this.lifetime.start < this.lifetime.end);

                    // No other node in the partition should be alive at this step
                    for other in partition.nodes.iter().filter(|it| it.node != this.node) {
                        assert!(
                            !other.lifetime.is_alive_at_step(step)
                                && other.lifetime.is_disjoint(&this.lifetime),
                            "Lifetime conflict @ step {}\n{:?}\n{:?}",
                            step,
                            this,
                            other
                        );
                    }

                    // This node should not be alive in another partition at the same step
                    for p in schema.by_partition.iter().filter(|it| it != &partition) {
                        if let Some(other) = p.find_node_alive_at_step(step) {
                            assert!(other.node != this.node);
                        }
                    }
                }
            }
        }

        // Verify schema usage
        let usage = schema.eval_usage(&symbol_values)?;
        assert!(usage >= EXPECTED_USAGE, "Usage {}, expected >= {}", usage, EXPECTED_USAGE);

        // Verify peak memory size
        let peak_memory_size = schema.eval_peak_memory_size(&symbol_values)?;
        assert_eq!(peak_memory_size, EXPECTED_PEAK_SIZE, "Peak memory size mismatch");

        Ok(())
    }
}
