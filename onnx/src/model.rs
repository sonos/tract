use std::collections::HashMap;

use tract_core::framework::{Framework, OpBuilder, OpRegister};
use tract_core::model::dsl::*;
use tract_core::model::{InletId, Model, OutletId};
use tract_core::*;

use crate::pb;

pub type OnnxOpRegister = OpRegister<pb::NodeProto>;

pub struct Onnx {
    pub op_register: OnnxOpRegister,
}

impl Framework<pb::NodeProto, pb::ModelProto> for Onnx {
    fn op_builder_for_name(&self, name: &str) -> Option<&OpBuilder<pb::NodeProto>> {
        self.op_register.get(name)
    }

    fn proto_model_for_read(&self, r: &mut std::io::Read) -> TractResult<pb::ModelProto> {
        Ok(::protobuf::parse_from_reader(r).map_err(|e| format!("{:?}", e))?)
    }

    fn model_for_proto_model(&self, proto: &pb::ModelProto) -> TractResult<InferenceModel> {
        let mut model = Model::default();
        let graph = proto.get_graph();
        let mut initializers: HashMap<&str, Tensor> = graph
            .get_initializer()
            .iter()
            .map(|init| Ok((init.get_name(), init.tractify()?)))
            .collect::<TractResult<_>>()?;
        let mut outlets_by_name = HashMap::<String, OutletId>::new();
        for input in graph.get_input().iter() {
            if let Some(init) = initializers.remove(input.get_name()) {
                let id = model.add_const(input.get_name().to_owned(), init.into())?;
                outlets_by_name.insert(input.get_name().to_owned(), OutletId::new(id, 0));
            } else {
                let fact = input.get_field_type().get_tensor_type().tractify()?;
                let id = model.add_source_fact(input.get_name().to_owned(), fact)?;
                outlets_by_name.insert(input.get_name().to_owned(), OutletId::new(id, 0));
            }
        }
        for pbnode in graph.get_node().iter() {
            let name = if pbnode.get_name() != "" {
                pbnode.get_name().to_string()
            } else if pbnode.get_output().len() > 0 && pbnode.get_output()[0] != "" {
                pbnode.get_output()[0].to_owned()
            } else {
                format!("{}-{}", model.nodes().len(), pbnode.get_op_type())
            };
            trace!("Creating node {}", name);
            let facts = (0..pbnode.get_output().len()).map(|_| TensorFact::default()).collect();
            trace!("  outputs {:?}", pbnode.get_output());
            let id = model.add_node(name, self.build_op(pbnode.get_op_type(), pbnode)?, facts)?;
            for (ix, output) in pbnode.get_output().iter().enumerate() {
                outlets_by_name.insert(output.to_owned(), OutletId::new(id, ix));
            }
            for (ix, input) in pbnode.get_input().iter().enumerate() {
                model.add_edge(outlets_by_name[&*input], InletId::new(id, ix))?;
            }
        }
        let mut outputs = vec![];
        for output in graph.get_output().iter() {
            let fact = output.get_field_type().get_tensor_type().tractify()?;
            outputs.push(outlets_by_name[output.get_name()]);
            model.set_fact(outlets_by_name[output.get_name()], fact)?;
        }
        model.set_outputs_outlets(&outputs)?;
        Ok(model)
    }
}
