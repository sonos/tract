use std::convert::TryInto;

use std::collections::HashMap;

use tract_core::internal::*;

use crate::pb;

#[derive(Clone)]
pub struct ParsingContext<'a> {
    pub framework: &'a Onnx,
    pub model: &'a pb::ModelProto,
    pub parent_graphs: Vec<&'a pb::GraphProto>,
}

#[derive(Clone, Debug)]
pub struct ParseResult {
    pub model: InferenceModel,
    pub unresolved_inputs: Vec<String>,
    pub outlets_by_name: HashMap<String, OutletId>,
}

impl<'a> ParsingContext<'a> {
    pub fn parse_graph(&self, graph: &pb::GraphProto) -> TractResult<ParseResult> {
        let mut ctx = self.clone();
        ctx.parent_graphs.push(graph);
        let mut model = InferenceModel::default();
        let mut unresolved_inputs = vec![];
        let mut closures_to_wire = vec![];
        let mut initializers: HashMap<&str, Tensor> = graph
            .get_initializer()
            .iter()
            .map(|init| Ok((init.get_name(), init.try_into()?)))
            .collect::<TractResult<_>>()?;
        for (k, v) in initializers.iter() {
            trace!("Initializer: {} {:?}", k, v);
        }
        let mut outlets_by_name = HashMap::<String, OutletId>::new();
        for input in graph.get_input().iter() {
            if let Some(init) = initializers.remove(input.get_name()) {
                trace!("Input: {} initialized by {:?}", input.get_name(), init);
                let id = model.add_const(input.get_name().to_owned(), init)?;
                outlets_by_name.insert(input.get_name().to_owned(), id);
            } else {
                let fact = input.get_field_type().get_tensor_type().try_into()?;
                trace!("Input: {} is a source ({:?})", input.get_name(), fact);
                let id = model.add_source(input.get_name(), fact)?;
                outlets_by_name.insert(input.get_name().to_owned(), OutletId::new(id, 0));
            }
        }
        for (name, t) in initializers.into_iter() {
            let id = model.add_const(name, t)?;
            outlets_by_name.insert(name.to_string(), id);
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
            let facts = pbnode
                .get_output()
                .iter()
                .filter(|s| !s.is_empty())
                .map(|_| TensorFact::default())
                .collect();
            trace!("  outputs {:?}", pbnode.get_output());
            let (op, closures) = match self.framework.op_register.0.get(pbnode.get_op_type()) {
                Some(builder) => (builder)(&ctx, pbnode)?,
                None => (
                    tract_core::ops::unimpl::UnimplementedOp::new(
                        pbnode.get_op_type(),
                        format!("{:?}", pbnode),
                    )
                    .into(),
                    vec![],
                ),
            };
            let id = model.add_node(name, op, facts)?;
            for (ix, output) in pbnode.get_output().iter().filter(|s| !s.is_empty()).enumerate() {
                outlets_by_name.insert(output.to_owned(), OutletId::new(id, ix));
            }
            for closure in closures {
                trace!("Node {} closes on {}", model.nodes()[id], closure);
                closures_to_wire.push((id, closure))
            }
        }
        for (id, pbnode) in graph.get_node().iter().enumerate() {
            for (ix, input) in pbnode.get_input().iter().filter(|s| !s.is_empty()).enumerate() {
                if !outlets_by_name.contains_key(&*input) {
                    let id = model.add_source_default(input.clone())?;
                    unresolved_inputs.push(input.to_string());
                    outlets_by_name.insert(input.to_string(), OutletId::new(id, 0));
                }
                let outlet = outlets_by_name[&*input];
                model.add_edge(outlet, InletId::new(id + consts, ix))?;
            }
        }
        for (id, closure) in closures_to_wire {
            if !outlets_by_name.contains_key(&*closure) {
                let id = model.add_source_default(closure.clone())?;
                unresolved_inputs.push(closure.to_string());
                outlets_by_name.insert(closure.to_string(), OutletId::new(id, 0));
            }
            let outlet = outlets_by_name[&*closure];
            let ix = model.nodes()[id].inputs.len();
            model.add_edge(outlet, InletId::new(id, ix))?;
        }
        let mut outputs = vec![];
        for output in graph.get_output().iter() {
            let fact = output.get_field_type().get_tensor_type().try_into()?;
            outputs.push(outlets_by_name[output.get_name()]);
            model.set_outlet_fact(outlets_by_name[output.get_name()], fact)?;
        }
        model.set_output_outlets(&outputs)?;
        let result = ParseResult { model, unresolved_inputs, outlets_by_name };
        Ok(result)
    }
}

#[derive(Clone, Default)]
pub struct OnnxOpRegister(
    pub  HashMap<
        String,
        fn(
            &ParsingContext,
            node: &pb::NodeProto,
        ) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)>,
    >,
);

impl OnnxOpRegister {
    pub fn insert(
        &mut self,
        s: &'static str,
        builder: fn(
            &ParsingContext,
            node: &pb::NodeProto,
        ) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)>,
    ) {
        self.0.insert(s.into(), builder);
    }
}

#[derive(Clone, Default)]
pub struct Onnx {
    pub op_register: OnnxOpRegister,
}

impl Onnx {
    pub fn parse(&self, proto: &pb::ModelProto) -> TractResult<ParseResult> {
        let graph = proto.get_graph();
        let ctx = ParsingContext { framework: self, model: proto, parent_graphs: vec![] };
        ctx.parse_graph(&graph)
    }
}

impl Framework<pb::ModelProto> for Onnx {
    fn proto_model_for_read(&self, r: &mut dyn std::io::Read) -> TractResult<pb::ModelProto> {
        Ok(::protobuf::parse_from_reader(r).map_err(|e| format!("{:?}", e))?)
    }

    fn model_for_proto_model(&self, proto: &pb::ModelProto) -> TractResult<InferenceModel> {
        let ParseResult { model, unresolved_inputs, .. } = self.parse(proto)?;
        if unresolved_inputs.len() > 0 {
            bail!("Could not resolve inputs at top-level: {:?}", unresolved_inputs)
        }
        Ok(model)
    }
}
