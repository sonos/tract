use std::convert::TryInto;

use std::collections::HashMap;

use tract_core::internal::*;

use crate::pb;

#[derive(Clone)]
pub struct ParsingContext<'a> {
    pub framework: &'a Onnx,
    pub model: &'a pb::ModelProto,
    pub parent_graphs: Vec<&'a pb::GraphProto>
}

impl<'a> ParsingContext<'a> {
    pub fn parse_graph(&self, graph: &pb::GraphProto) -> TractResult<InferenceModel> {
        let mut ctx = self.clone();
        ctx.parent_graphs.push(graph);
        let mut model = Model::default();
        let mut initializers: HashMap<&str, Tensor> = graph
            .get_initializer()
            .iter()
            .map(|init| Ok((init.get_name(), init.try_into()?)))
            .collect::<TractResult<_>>()?;
        let mut outlets_by_name = HashMap::<String, OutletId>::new();
        for input in graph.get_input().iter() {
            if let Some(init) = initializers.remove(input.get_name()) {
                let id = model.add_const(input.get_name().to_owned(), init)?;
                outlets_by_name.insert(input.get_name().to_owned(), OutletId::new(id, 0));
            } else {
                let fact = input.get_field_type().get_tensor_type().try_into()?;
                let id = model.add_source(input.get_name(), fact)?;
                outlets_by_name.insert(input.get_name().to_owned(), OutletId::new(id, 0));
            }
        }
        let consts = model.nodes().len();
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
            let op = match self.framework.op_register.0.get(pbnode.get_op_type()) {
                Some(builder) => (builder)(&ctx, pbnode)?,
                None => tract_core::ops::unimpl::UnimplementedOp::new(pbnode.get_op_type(),
                            format!("{:?}", pbnode)).into(),
            };
            let id = model.add_node(name, op, facts)?;
            for (ix, output) in pbnode.get_output().iter().enumerate() {
                outlets_by_name.insert(output.to_owned(), OutletId::new(id, ix));
            }
        }
        for (id, pbnode) in graph.get_node().iter().enumerate() {
            for (ix, input) in pbnode.get_input().iter().filter(|s| s.len() > 0).enumerate() {
                model.add_edge(outlets_by_name[&*input], InletId::new(id + consts, ix))?;
            }
        }
        let mut outputs = vec![];
        for output in graph.get_output().iter() {
            let fact = output.get_field_type().get_tensor_type().try_into()?;
            outputs.push(outlets_by_name[output.get_name()]);
            model.set_outlet_fact(outlets_by_name[output.get_name()], fact)?;
        }
        model.set_output_outlets(&outputs)?;
        Ok(model)
    }
}

#[derive(Clone, Default)]
pub struct OnnxOpRegister(pub HashMap<String, fn(&ParsingContext, node: &pb::NodeProto) -> TractResult<Box<InferenceOp>>>);

impl OnnxOpRegister {
    pub fn insert(&mut self, s: &'static str, builder: fn(&ParsingContext, node: &pb::NodeProto) -> TractResult<Box<InferenceOp>>) {
        self.0.insert(s.into(), builder);
    }
}

#[derive(Clone, Default)]
pub struct Onnx {
    pub op_register: OnnxOpRegister,
}

impl Framework<pb::ModelProto> for Onnx {
    fn proto_model_for_read(&self, r: &mut std::io::Read) -> TractResult<pb::ModelProto> {
        Ok(::protobuf::parse_from_reader(r).map_err(|e| format!("{:?}", e))?)
    }

    fn model_for_proto_model(&self, proto: &pb::ModelProto) -> TractResult<InferenceModel> {
        let graph = proto.get_graph();
        let ctx = ParsingContext { framework: self, model: proto, parent_graphs: vec!() };
        ctx.parse_graph(&graph)
    }
}
