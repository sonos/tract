use crate::tfpb::graph::GraphDef;
use crate::tfpb::node_def::NodeDef;
use tract_core::framework::{Framework, OpRegister, OpBuilder};
use tract_core::model::{InletId, Model, OutletId};
use tract_core::TractResult;

pub type TfOpRegister = OpRegister<NodeDef>;

pub struct Tensorflow {
    pub op_register: TfOpRegister
}

impl Framework<NodeDef, GraphDef> for Tensorflow {
    fn op_builder_for_name(&self, name: &str) -> Option<&OpBuilder<NodeDef>> {
        self.op_register.get(name)
    }
    fn proto_model_for_read(&self, r: &mut std::io::Read) -> TractResult<GraphDef> {
        Ok(::protobuf::parse_from_reader::<GraphDef>(r).map_err(|e| format!("{:?}", e))?)
    }

    fn model_for_proto_model(&self, graph: &GraphDef) -> TractResult<Model> {
        let mut model = Model::default().with_norm_optims(Some(crate::optim::normalization()));
        for pbnode in graph.get_node().iter() {
            let name = pbnode.get_name().to_string();
            let node_id = model.add_node(
                name.clone(),
                self
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
}
