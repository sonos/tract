use std::convert::TryInto;

use std::collections::HashMap;

use tract_core::internal::*;
use tract_core::infer::*;

use crate::pb;
use prost::Message;

pub fn optional_inputs(pb: &pb::NodeProto) -> impl Iterator<Item = Option<usize>> + '_ {
    let mut real_input = 0;
    (0..).map(move |i| {
        if pb.input.get(i).filter(|s| !s.is_empty()).is_some() {
            real_input += 1;
            Some(real_input - 1)
        } else {
            None
        }
    })
}

pub fn optional_outputs(pb: &pb::NodeProto) -> impl Iterator<Item = Option<usize>> + '_ {
    let mut real_input = 0;
    (0..).map(move |i| {
        if pb.output.get(i).filter(|s| !s.is_empty()).is_some() {
            real_input += 1;
            Some(real_input - 1)
        } else {
            None
        }
    })
}

#[derive(Clone)]
pub struct ParsingContext<'a> {
    pub onnx_operator_set_version: i64,
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
            .initializer
            .iter()
            .map(|init| Ok((&*init.name, init.try_into()?)))
            .collect::<TractResult<_>>()?;
        for (k, v) in initializers.iter() {
            trace!("Initializer: {} {:?}", k, v);
        }
        let mut outlets_by_name = HashMap::<String, OutletId>::new();
        for input in graph.input.iter() {
            if let Some(init) = initializers.remove(&*input.name) {
                trace!("Input: {} initialized by {:?}", input.name, init);
                let id = model.add_const(input.name.to_owned(), init)?;
                outlets_by_name.insert(input.name.to_owned(), id);
            } else {
                let fact = input.r#type.as_ref().unwrap().value.as_ref().unwrap();
                #[allow(irrefutable_let_patterns)]
                let fact: InferenceFact = if let pb::type_proto::Value::TensorType(fact) = fact {
                    fact.try_into()?
                } else {
                    bail!("Can not parse tensor type");
                };
                trace!("Input: {} is a source ({:?})", input.name, fact);
                let id = model.add_source(&*input.name, fact)?;
                outlets_by_name.insert(input.name.to_owned(), id);
            }
        }
        for output in graph.output.iter() {
            trace!("Model output: {:?}", output);
        }
        for (name, t) in initializers.into_iter() {
            let id = model.add_const(name, t)?;
            outlets_by_name.insert(name.to_string(), id);
        }
        let consts = model.nodes().len();
        for pbnode in graph.node.iter() {
            let name = if pbnode.name != "" {
                pbnode.name.to_string()
            } else if pbnode.output.len() > 0 && pbnode.output[0] != "" {
                pbnode.output[0].to_owned()
            } else {
                format!("{}-{}", model.nodes().len(), pbnode.op_type)
            };
            trace!("Creating node {}", name);
            let facts = pbnode
                .output
                .iter()
                .filter(|s| !s.is_empty())
                .map(|_| InferenceFact::default())
                .collect();
            trace!("  outputs {:?}", pbnode.output);
            let (op, closures) = match self.framework.op_register.0.get(&pbnode.op_type) {
                Some(builder) => (builder)(&ctx, pbnode)?,
                None => (
                    tract_core::ops::unimpl::UnimplementedOp::new(
                        pbnode.output.len(),
                        &*pbnode.op_type,
                        format!("{:?}", pbnode),
                    )
                    .into(),
                    vec![],
                ),
            };
            let id = model.add_node(name, op, facts)?;
            for (ix, output) in pbnode.output.iter().filter(|s| !s.is_empty()).enumerate() {
                outlets_by_name.insert(output.to_owned(), OutletId::new(id, ix));
                model.set_outlet_label(OutletId::new(id, ix), output.to_owned());
            }
            for closure in closures {
                trace!("Node {} closes on {}", model.nodes()[id], closure);
                closures_to_wire.push((id, closure))
            }
        }
        for (id, pbnode) in graph.node.iter().enumerate() {
            for (ix, input) in pbnode.input.iter().filter(|s| !s.is_empty()).enumerate() {
                if !outlets_by_name.contains_key(&*input) {
                    let id = model.add_source(input.clone(), InferenceFact::default())?;
                    unresolved_inputs.push(input.to_string());
                    outlets_by_name.insert(input.to_string(), id);
                }
                let outlet = outlets_by_name[&*input];
                model.add_edge(outlet, InletId::new(id + consts, ix))?;
            }
        }
        for (id, closure) in closures_to_wire {
            if !outlets_by_name.contains_key(&*closure) {
                let id = model.add_source(closure.clone(), InferenceFact::default())?;
                unresolved_inputs.push(closure.to_string());
                outlets_by_name.insert(closure.to_string(), id);
            }
            let outlet = outlets_by_name[&*closure];
            let ix = model.nodes()[id].inputs.len();
            model.add_edge(outlet, InletId::new(id, ix))?;
        }
        let mut outputs = vec![];
        for output in graph.output.iter() {
            let fact = output.r#type.as_ref().unwrap().value.as_ref().unwrap();
            #[allow(irrefutable_let_patterns)]
            let fact = if let pb::type_proto::Value::TensorType(fact) = fact {
                fact.try_into()?
            } else {
                bail!("Can not parse tensor type");
            };
            outputs.push(outlets_by_name[&*output.name]);
            model.set_outlet_fact(outlets_by_name[&*output.name], fact)?;
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
        let onnx_operator_set_version =
            proto.opset_import.iter().find(|import| import.domain == "").unwrap().version;
        let graph = &proto.graph;
        let ctx = ParsingContext {
            framework: self,
            model: proto,
            parent_graphs: vec![],
            onnx_operator_set_version,
        };
        ctx.parse_graph(graph.as_ref().unwrap())
    }
}

impl Framework<pb::ModelProto> for Onnx {
    fn proto_model_for_read(&self, r: &mut dyn std::io::Read) -> TractResult<pb::ModelProto> {
        let mut v = vec![];
        r.read_to_end(&mut v)?;
        let b = bytes::Bytes::from(v);
        Ok(crate::pb::ModelProto::decode(b).map_err(|e| format!("{:?}", e))?)
    }

    fn model_for_proto_model(&self, proto: &pb::ModelProto) -> TractResult<InferenceModel> {
        let ParseResult { model, unresolved_inputs, .. } = self.parse(proto)?;
        if unresolved_inputs.len() > 0 {
            bail!("Could not resolve inputs at top-level: {:?}", unresolved_inputs)
        }
        Ok(model)
    }
}
