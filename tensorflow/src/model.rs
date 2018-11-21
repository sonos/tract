use std::{fs, path};
use std::sync::Arc;

use tfpb::graph::GraphDef;
use tract_core::model::{InletId, Model, OutletId};
use tract_core::{ToTract, TractResult, Tractify};

/// Load a SharedTensorflow protobul model from a file.
pub fn for_path<P: AsRef<path::Path>>(p: P) -> TractResult<Model> {
    for_reader(fs::File::open(p)?)
}

/// Load a Tract model from a reader.
pub fn for_reader<R: ::std::io::Read>(r: R) -> TractResult<Model> {
    graphdef_for_reader(r)?.tractify()
}

/// Load a SharedTensorflow protobuf graph def from a reader.
pub fn graphdef_for_reader<R: ::std::io::Read>(mut r: R) -> TractResult<GraphDef> {
    Ok(::protobuf::parse_from_reader::<GraphDef>(&mut r).map_err(|e| format!("{:?}", e))?)
}

/// Load a SharedTensorflow protobuf graph def from a path
pub fn graphdef_for_path<P: AsRef<path::Path>>(p: P) -> TractResult<GraphDef> {
    graphdef_for_reader(fs::File::open(p)?)
}

pub fn optimize(model: Model) -> TractResult<Model> {
    let model = model.into_optimized()?;
    model.into_optimized()
}

impl Tractify<GraphDef> for Model {
    fn tractify(graph: &GraphDef) -> TractResult<Model> {
        let mut model = Model::default().with_context(Arc::new(::optim::TensorflowContext));
        let op_builder = ::ops::OpBuilder::new();
        for pbnode in graph.get_node().iter() {
            let name = pbnode.get_name().to_string();
            let node_id = model.add_node(
                name.clone(),
                op_builder
                    .build(pbnode)
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
                    (
                        splits[0],
                        if splits.len() > 1 {
                            splits[1].parse::<usize>()?
                        } else {
                            0
                        },
                    )
                };
                let prec = model.node_by_name(input.0)?.id;
                model.add_edge(OutletId::new(prec, input.1), InletId::new(node_id, ix))?;
            }
        }
        Ok(model)
    }
}
