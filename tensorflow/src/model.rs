use std::{fs, path};

use tract_core::model::{InletId, Model, OutletId};
use tract_core::{TfdFrom, TfdResult, ToTfd};
use tfpb::graph::GraphDef;

/// Load a Tensorflow protobul model from a file.
pub fn for_path<P: AsRef<path::Path>>(p: P) -> TfdResult<Model> {
    for_reader(fs::File::open(p)?)
}

/// Load a Tfdeploy model from a reader.
pub fn for_reader<R: ::std::io::Read>(r: R) -> TfdResult<Model> {
    graphdef_for_reader(r)?.to_tfd()
}

/// Load a Tensorflow protobuf graph def from a reader.
pub fn graphdef_for_reader<R: ::std::io::Read>(mut r: R) -> TfdResult<GraphDef> {
    Ok(::protobuf::parse_from_reader::<GraphDef>(&mut r).map_err(|e| format!("{:?}", e))?)
}

/// Load a Tensorflow protobuf graph def from a path
pub fn graphdef_for_path<P: AsRef<path::Path>>(p: P) -> TfdResult<GraphDef> {
    graphdef_for_reader(fs::File::open(p)?)
}

pub fn optimize(model: Model) -> TfdResult<Model> {
    let model = model.into_optimized()?;
    let model = ::optim::untf_convos(model)?;
    model.into_optimized()
}

impl TfdFrom<GraphDef> for Model {
    fn tfd_from(graph: &GraphDef) -> TfdResult<Model> {
        let mut model = Model::default();
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
