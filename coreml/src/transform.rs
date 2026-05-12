//! `CoremlTransform`: walk a `TypedModel`, partition translatable ops into
//! maximal connected subgraphs, and replace each subgraph with a single
//! `CoremlOp` that owns one MLPackage covering the whole region.
//!
//! Phase 2 architecture (the per-op Phase 1 first cut now lives only as the
//! degenerate size-1 subgraph case). See `crate::fusion` for the subgraph
//! identification + materialisation; this module just orchestrates the
//! source → target rewrite.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use anyhow::{Context, Result};
use objc2_core_ml::MLComputeUnits;
use tract_core::internal::*;
use tract_core::transform::ModelTransform;

use crate::context::CoremlContext;
use crate::coreml_op::CoremlOp;
use crate::fusion::{self, Subgraph, SubgraphIO};
use crate::ops::conv;

/// Configuration for the Coreml transform.
#[derive(Debug, Clone)]
pub struct CoremlTransform {
    /// Compute units to request from Core ML at MLModel load time.
    pub compute_units: MLComputeUnits,
}

impl Default for CoremlTransform {
    fn default() -> Self {
        Self { compute_units: MLComputeUnits::All }
    }
}

impl ModelTransform for CoremlTransform {
    fn name(&self) -> StaticName {
        "coreml-transform".into()
    }

    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        let translated = self.rewrite_with_subgraphs(model)?;
        *model = translated;
        Ok(())
    }
}

impl CoremlTransform {
    /// Build a new typed model from `source` by:
    /// 1. Identifying maximal connected subgraphs of translatable nodes.
    /// 2. For each subgraph: materialise an MLPackage, load it, and produce a
    ///    `CoremlOp` node carrying the right I/O wiring.
    /// 3. Walk `source` in topological order copying nodes to the target,
    ///    replacing whole subgraphs with their `CoremlOp` and skipping the
    ///    constants that got absorbed into the MLPackage.
    fn rewrite_with_subgraphs(&self, source: &TypedModel) -> TractResult<TypedModel> {
        // 1. Identify subgraphs.
        let subgraphs = fusion::identify_subgraphs(source)?;
        log::debug!("CoremlTransform: identified {} translatable subgraph(s)", subgraphs.len());
        if std::env::var("TRACT_COREML_DEBUG_SUBGRAPHS").is_ok() {
            eprintln!(
                "[CoremlTransform] {} subgraphs ({} total nodes)",
                subgraphs.len(),
                source.nodes.len()
            );
            for (i, sg) in subgraphs.iter().enumerate() {
                let names: Vec<&str> =
                    sg.nodes.iter().map(|&n| source.nodes[n].name.as_str()).collect();
                eprintln!(
                    "  sg[{i}]: {} node(s) {:?}  ext_in={} ext_out={}",
                    sg.nodes.len(),
                    names,
                    sg.external_inputs.len(),
                    sg.external_outputs.len()
                );
                for (j, ein) in sg.external_inputs.iter().enumerate() {
                    eprintln!(
                        "    ext_in[{j}] = {ein:?} from node {} ({})",
                        ein.node, source.nodes[ein.node].name
                    );
                }
            }
        }

        // 2. Pre-build each subgraph's MLPackage + CoremlOp.
        let mut materialised: Vec<MaterialisedSubgraph> = Vec::with_capacity(subgraphs.len());
        for sg in &subgraphs {
            materialised.push(self.materialise_subgraph(source, sg)?);
        }

        // 3. For fast lookup during the walk, build per-source-node-id maps:
        //    - subgraph_owner[node_id] = index into materialised, if member
        //    - the "first" (smallest topo index) member of each subgraph is
        //      where we wire the CoremlOp into the target model
        let mut subgraph_owner: HashMap<usize, usize> = HashMap::new();
        for (i, m) in materialised.iter().enumerate() {
            for &n in &m.node_set {
                subgraph_owner.insert(n, i);
            }
        }

        // The first node we encounter for each subgraph (in topo order)
        // triggers MLPackage wiring; later members are skipped.
        let mut subgraph_wired: Vec<bool> = vec![false; materialised.len()];

        // 4. Walk source in topo order, building target. Subgraphs and non-
        //    subgraph nodes that aren't ready (some input not yet mapped) are
        //    deferred to a follow-up pass — necessary because subgraphs may
        //    contain non-adjacent members in topo order, so the wire-trigger
        //    moment for a subgraph isn't fixed.
        let mut target = TypedModel { symbols: source.symbols.clone(), ..Default::default() };
        let mut mapping: HashMap<OutletId, OutletId> = HashMap::new();

        // Pre-process: wire every model-input Source into the target,
        // regardless of whether it has live consumers. `source.eval_order()`
        // excludes dead-input Sources (e.g. past_key_values inputs at
        // past_len=0 that declutter eliminated), but `source.input_outlets()`
        // still lists them — and step 5 below expects every input to map to
        // a target outlet. Without this, transform panics with "model input
        // unmapped" on LLM ONNX exports that bind ≥1 zero-size past-cache
        // input. Surfaced on SmolLM2-135M (60 dead KV-cache inputs).
        for &input_outlet in source.input_outlets()?.iter() {
            if mapping.contains_key(&input_outlet) {
                continue;
            }
            let n = &source.nodes[input_outlet.node];
            let new_outlets =
                target.wire_node(n.name.clone(), n.op.clone(), &[]).with_context(|| {
                    format!("wiring pre-mapped model input {} (outlet {:?})", n.name, input_outlet)
                })?;
            for (slot, &new) in new_outlets.iter().enumerate() {
                mapping.insert(OutletId::new(input_outlet.node, slot), new);
            }
        }

        let mut to_process: Vec<usize> = source.eval_order()?;

        loop {
            let mut still_pending: Vec<usize> = Vec::new();
            let mut progress = false;

            for n_id in to_process.iter().copied() {
                // Is this node part of an absorbed subgraph?
                if let Some(&sg_idx) = subgraph_owner.get(&n_id) {
                    if subgraph_wired[sg_idx] {
                        // Already done; nothing to do for this member.
                        continue;
                    }
                    let m = &materialised[sg_idx];
                    if !m.io.source_outlets.iter().all(|o| mapping.contains_key(o)) {
                        // Not ready yet — some external input still pending.
                        still_pending.push(n_id);
                        continue;
                    }
                    let inputs_in_target: TVec<OutletId> =
                        m.io.source_outlets.iter().map(|src| mapping[src]).collect();
                    let outlets = target.wire_node(
                        format!("coreml_subgraph_{sg_idx}"),
                        m.op.clone(),
                        &inputs_in_target,
                    )?;
                    for (slot, src_outlet) in m.io.source_outputs.iter().enumerate() {
                        mapping.insert(*src_outlet, outlets[slot]);
                    }
                    subgraph_wired[sg_idx] = true;
                    progress = true;
                    continue;
                }

                let node = &source.nodes[n_id];

                // Already wired by the pre-processing pass (model-input
                // Sources) — skip to avoid double-wiring.
                if mapping.contains_key(&OutletId::new(n_id, 0)) {
                    progress = true;
                    continue;
                }

                if is_only_consumed_by_subgraphs(source, n_id, &subgraph_owner) {
                    // Const-like dead node — don't copy.
                    continue;
                }

                if !node.inputs.iter().all(|o| mapping.contains_key(o)) {
                    still_pending.push(n_id);
                    continue;
                }

                let mapped_inputs: TVec<OutletId> =
                    node.inputs.iter().map(|o| mapping[o]).collect();
                let new_outlets =
                    target.wire_node(node.name.clone(), node.op.clone(), &mapped_inputs)?;
                for (slot, &new) in new_outlets.iter().enumerate() {
                    mapping.insert(OutletId::new(n_id, slot), new);
                }
                progress = true;
            }

            if still_pending.is_empty() {
                break;
            }
            if !progress {
                if std::env::var("TRACT_COREML_DEBUG_SUBGRAPHS").is_ok() {
                    eprintln!("[CoremlTransform] STUCK; {} pending:", still_pending.len());
                    for &n_id in still_pending.iter().take(20) {
                        let n = &source.nodes[n_id];
                        if let Some(&sg_idx) = subgraph_owner.get(&n_id) {
                            let m = &materialised[sg_idx];
                            let missing: Vec<_> =
                                m.io.source_outlets
                                    .iter()
                                    .filter(|o| !mapping.contains_key(o))
                                    .collect();
                            eprintln!(
                                "  sg-member {n_id} {} (sg={sg_idx}, wired={}); missing {} ext_in: {:?}",
                                n.name,
                                subgraph_wired[sg_idx],
                                missing.len(),
                                missing
                            );
                        } else {
                            let missing: Vec<_> =
                                n.inputs.iter().filter(|o| !mapping.contains_key(o)).collect();
                            eprintln!(
                                "  non-sg {n_id} {} (op={}); missing {} input: {:?}",
                                n.name,
                                n.op.name(),
                                missing.len(),
                                missing
                            );
                        }
                    }
                }
                bail!(
                    "subgraph wiring stuck: {} pending node(s); cyclic dep or boundary bug",
                    still_pending.len()
                );
            }
            to_process = still_pending;
        }

        // 5. Set target inputs + outputs (mirrors tract-core's `Translate`
        //    trait — wire_node-ing a Source op doesn't auto-register it as a
        //    model input; we have to do it explicitly here).
        target.inputs = source
            .input_outlets()?
            .iter()
            .map(|i| *mapping.get(i).expect("model input unmapped"))
            .collect();
        let new_outputs: Vec<OutletId> = source
            .outputs
            .iter()
            .map(|o| *mapping.get(o).expect("model output unmapped"))
            .collect();
        target.select_output_outlets(&new_outputs)?;

        Ok(target)
    }

    fn materialise_subgraph(
        &self,
        source: &TypedModel,
        subgraph: &Subgraph,
    ) -> Result<MaterialisedSubgraph> {
        // Build the artifacts in memory so we can hash them before deciding
        // whether to write the .mlpackage to disk + recompile.
        let (io, model_bytes, weight_bytes) = fusion::build_subgraph_artifacts(source, subgraph)?;

        let pkg_path = conv::next_temp_pkg_path();
        let model_bytes_for_write = model_bytes.clone();
        let weight_bytes_for_write = weight_bytes.clone();
        let (ctx, status) = CoremlContext::load_mlpackage_cached(
            &pkg_path,
            &model_bytes,
            &weight_bytes,
            self.compute_units,
            move |path| {
                crate::mlpackage::write(path, &model_bytes_for_write, &weight_bytes_for_write)
                    .map_err(|e| anyhow::anyhow!("mlpackage::write({path:?}) failed: {e}"))
            },
        )?;
        if std::env::var("TRACT_COREML_DEBUG_SUBGRAPHS").is_ok() {
            let key = crate::compile_cache::CacheKey::compute(&model_bytes, &weight_bytes);
            eprintln!(
                "[CoremlTransform] subgraph cache: {status:?} key={}.. ({} nodes; mlmodel {} B; weight {} B)",
                &key.digest[..16],
                subgraph.nodes.len(),
                model_bytes.len(),
                weight_bytes.len()
            );
        }
        let ctx = Arc::new(ctx);
        let op = CoremlOp {
            context: ctx,
            input_names: io.input_names.clone(),
            output_names: io.output_names.clone(),
            output_facts: io.output_facts.clone(),
            coreml_input_shapes: io.coreml_input_shapes.clone(),
            coreml_input_dtypes: io.coreml_input_dtypes.clone(),
            coreml_output_shapes: io.coreml_output_shapes.clone(),
        };
        let node_set: HashSet<usize> = subgraph.nodes.iter().copied().collect();
        Ok(MaterialisedSubgraph { op, io, node_set })
    }
}

struct MaterialisedSubgraph {
    op: CoremlOp,
    io: SubgraphIO,
    node_set: HashSet<usize>,
}

/// True if `n_id` is a const-like node (Const, or `Cast(Const)`) whose every
/// consumer is inside *some* subgraph — i.e. its value is fully absorbed into
/// MLPackages and no surviving node needs it. Only const-like nodes are eligible
/// for this dead-code elimination; sources and other non-const nodes are
/// always preserved.
fn is_only_consumed_by_subgraphs(
    source: &TypedModel,
    n_id: usize,
    subgraph_owner: &HashMap<usize, usize>,
) -> bool {
    use tract_core::ops::cast::Cast;
    use tract_core::ops::konst::Const;

    let n = &source.nodes[n_id];

    // Only Const / Cast(Const) get pruned. Other non-translatable nodes (Source,
    // Add, ReLU, etc.) stay even if their consumers all became subgraphs —
    // they may have side effects or be load-bearing for the model's structure.
    let is_const_like = if n.op_as::<Const>().is_some() {
        true
    } else if n.op_as::<Cast>().is_some() && n.inputs.len() == 1 {
        source.nodes[n.inputs[0].node].op_as::<Const>().is_some()
    } else {
        false
    };
    if !is_const_like {
        return false;
    }

    // Model outputs keep the node alive.
    for slot in 0..n.outputs.len() {
        let outlet = OutletId::new(n_id, slot);
        if source.outputs.contains(&outlet) {
            return false;
        }
    }

    let mut had_consumer = false;
    for other in &source.nodes {
        for input in &other.inputs {
            if input.node == n_id {
                had_consumer = true;
                if !subgraph_owner.contains_key(&other.id) {
                    return false;
                }
            }
        }
    }
    had_consumer
}
