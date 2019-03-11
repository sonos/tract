use std::sync::Arc;

use crate::tfpb::graph::GraphDef;
use tract_core::model::{InletId, Model, OutletId};
use tract_core::TractResult;

pub fn load(r: &mut std::io::Read) -> TractResult<GraphDef> {
    Ok(::protobuf::parse_from_reader::<GraphDef>(r).map_err(|e| format!("{:?}", e))?)
}

pub fn build(graph: &GraphDef, framework: &crate::Tensorflow) -> TractResult<Model> {
    let mut model = Model::default().with_context(Arc::new(crate::optim::TensorflowContext));
    for pbnode in graph.get_node().iter() {
        let name = pbnode.get_name().to_string();
        let node_id = model.add_node(
            name.clone(),
            framework
                .build_op(&*pbnode.get_op(), pbnode)
                .map_err(|e| format!("While building node {}, {}", name, e.description()))?,
        )?;

        // From the node_def.proto documentation:
        // Each input is "node:src_output" with "node" being a string name and
        // "src_output" indicating which output tensor to use from "node". If
        // "src_output" is 0 the ":0" suffix can be omitted. Regular inputs may
        // optionally be followed by control inputs that have the format "^node".
        for (ix, i) in pbnode.get_input().iter().enumerate() {
            let input: (&str, usize) = if i.starts_with("^") {
                (&i[1..], 0)
            } else {
                let splits: Vec<_> = i.splitn(2, ':').collect();
                (splits[0], if splits.len() > 1 { splits[1].parse::<usize>()? } else { 0 })
            };
            let prec = model.node_by_name(input.0)?.id;
            model.add_edge(OutletId::new(prec, input.1), InletId::new(node_id, ix))?;
        }
    }
    Ok(model)
}
