use crate::annotations::{Annotations, NodeQId};
use crate::model::Model;
use serde::Serialize;
use std::collections::HashMap;
use tract_core::internal::*;

#[derive(Clone, Debug, Default, Serialize)]
pub struct GraphPerfInfo {
    nodes: Vec<Node>,
    profiling_info: Option<ProfilingInfo>,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize)]
pub struct NodeQIdSer(pub Vec<(usize, String)>, pub usize);

#[derive(Clone, Debug, Serialize)]
pub struct Node {
    qualified_id: NodeQIdSer,
    op_name: String,
    node_name: String,

    #[serde(skip_serializing_if = "HashMap::is_empty")]
    cost: HashMap<String, String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    secs_per_iter: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ProfilingInfo {
    iterations: usize,
    secs_per_iter: f64,
}

impl GraphPerfInfo {
    pub fn from(model: &dyn Model, annotations: &Annotations) -> GraphPerfInfo {
        let nodes = annotations
            .tags
            .iter()
            .map(|(id, node)| Node {
                qualified_id: NodeQIdSer(id.0.iter().cloned().collect(), id.1),
                cost: node.cost.iter().map(|(k, v)| (format!("{k:?}"), format!("{v}"))).collect(),
                node_name: id.model(model).unwrap().node_name(id.1).to_string(),
                op_name: id.model(model).unwrap().node_op_name(id.1).to_string(),
                secs_per_iter: node.profile.map(|s| s.as_secs_f64()),
            })
            .collect();
        let profiling_info = annotations.profile_summary.as_ref().map(|summary| ProfilingInfo {
            secs_per_iter: summary.entire.as_secs_f64(),
            iterations: summary.iters,
        });
        GraphPerfInfo { nodes, profiling_info }
    }
}

// -- audit-json --

#[derive(Serialize)]
pub struct AuditModel {
    properties: HashMap<String, String>,
    assertions: Vec<String>,
    inputs: Vec<AuditModelIo>,
    outputs: Vec<AuditModelIo>,
    nodes: Vec<AuditNode>,
}

#[derive(Serialize)]
struct AuditModelIo {
    name: String,
    node: usize,
    slot: usize,
    fact: String,
}

#[derive(Serialize)]
struct AuditNode {
    id: usize,
    name: String,
    op: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    info: Vec<String>,
    inputs: Vec<AuditOutletRef>,
    outputs: Vec<AuditNodeOutput>,
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    cost: HashMap<String, String>,
}

#[derive(Serialize)]
struct AuditOutletRef {
    node: usize,
    slot: usize,
}

#[derive(Serialize)]
struct AuditNodeOutput {
    fact: String,
    successors: Vec<AuditInletRef>,
}

#[derive(Serialize)]
struct AuditInletRef {
    node: usize,
    slot: usize,
}

pub fn audit_json(
    model: &dyn Model,
    annotations: &Annotations,
    writer: impl std::io::Write,
) -> TractResult<()> {
    let properties: HashMap<String, String> =
        model.properties().iter().map(|(k, v)| (k.clone(), format!("{v:?}"))).collect();

    let scope = model.symbols();
    let assertions: Vec<String> = scope.all_assertions().iter().map(|a| format!("{a}")).collect();

    let inputs: Vec<AuditModelIo> = model
        .input_outlets()
        .iter()
        .map(|o| {
            Ok(AuditModelIo {
                name: model.node_name(o.node).to_string(),
                node: o.node,
                slot: o.slot,
                fact: model.outlet_fact_format(*o),
            })
        })
        .collect::<TractResult<_>>()?;

    let outputs: Vec<AuditModelIo> = model
        .output_outlets()
        .iter()
        .map(|o| {
            Ok(AuditModelIo {
                name: model.node_name(o.node).to_string(),
                node: o.node,
                slot: o.slot,
                fact: model.outlet_fact_format(*o),
            })
        })
        .collect::<TractResult<_>>()?;

    let nodes: Vec<AuditNode> = (0..model.nodes_len())
        .map(|id| {
            let op = model.node_op(id);
            let info = op.info().unwrap_or_default();
            let node_inputs: Vec<AuditOutletRef> = model
                .node_inputs(id)
                .iter()
                .map(|o| AuditOutletRef { node: o.node, slot: o.slot })
                .collect();
            let node_outputs: Vec<AuditNodeOutput> = (0..model.node_output_count(id))
                .map(|slot| {
                    let outlet = OutletId::new(id, slot);
                    let fact = model.outlet_fact_format(outlet);
                    let successors: Vec<AuditInletRef> = model
                        .outlet_successors(outlet)
                        .iter()
                        .map(|inlet| AuditInletRef { node: inlet.node, slot: inlet.slot })
                        .collect();
                    AuditNodeOutput { fact, successors }
                })
                .collect();
            let cost: HashMap<String, String> = annotations
                .tags
                .get(&NodeQId(tvec!(), id))
                .map(|tags| {
                    tags.cost.iter().map(|(k, v)| (format!("{k:?}"), format!("{v}"))).collect()
                })
                .unwrap_or_default();
            AuditNode {
                id,
                name: model.node_name(id).to_string(),
                op: model.node_op_name(id).to_string(),
                info,
                inputs: node_inputs,
                outputs: node_outputs,
                cost,
            }
        })
        .collect();

    let audit = AuditModel { properties, assertions, inputs, outputs, nodes };
    serde_json::to_writer_pretty(writer, &audit)?;
    Ok(())
}
