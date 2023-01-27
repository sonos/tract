use std::collections::HashMap;
use serde::Serialize;
use tract_core::internal::*;
use crate::annotations::Annotations;
use crate::model::Model;

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
                cost: node
                    .cost
                    .iter()
                    .map(|(k, v)| (format!("{k:?}"), format!("{v}")))
                    .collect(),
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
