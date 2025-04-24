use std::path::PathBuf;
use std::{fs, path};

use std::collections::HashMap;

use tract_hir::internal::*;
use tract_hir::prelude::tract_itertools::Itertools;

use crate::data_resolver::{self, ModelDataResolver};
use crate::pb::type_proto::Value;
use crate::pb::{self, TensorProto, TypeProto};
use crate::tensor::{load_tensor, translate_inference_fact};
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
    let mut real_output = 0;
    (0..).map(move |i| {
        if pb.output.get(i).filter(|s| !s.is_empty()).is_some() {
            real_output += 1;
            Some(real_output - 1)
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
    pub model_dir: Option<&'a str>,
    pub template: InferenceModel,
}

#[derive(Clone, Debug)]
pub struct ParseResult {
    pub model: InferenceModel,
    pub unresolved_inputs: Vec<String>,
    pub outlets_by_name: HashMap<String, OutletId>,
}

impl ParsingContext<'_> {
    pub fn load_tensor(&self, proto: &TensorProto) -> TractResult<Tensor> {
        load_tensor(&*self.framework.provider, proto, self.model_dir)
    }

    pub fn parse_graph(&self, graph: &pb::GraphProto) -> TractResult<ParseResult> {
        let mut ctx = self.clone();
        ctx.parent_graphs.push(graph);
        let mut model = self.template.clone();
        let mut unresolved_inputs = vec![];
        let mut closures_to_wire = vec![];
        trace!("trying to initialize initializers hashmap...");
        #[allow(unused_assignments)]
        let mut initializers: HashMap<&str, Tensor> = graph
            .initializer
            .iter()
            .map(|name| {
                let t = self.load_tensor(name)?;
                Ok((&*name.name, t))
            })
            .collect::<TractResult<_>>()?;
        for (k, v) in initializers.iter().sorted_by_key(|kv| kv.0) {
            trace!("Initializer: {k} {v:?}");
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
                    translate_inference_fact(&ctx, fact, true)
                        .with_context(|| format!("translating to fact: {fact:?}"))?
                } else {
                    bail!("Can not parse tensor type");
                };
                trace!("Input: {} is a source ({:?})", input.name, fact);
                let id = model.add_source(&*input.name, fact)?;
                outlets_by_name.insert(input.name.to_owned(), id);
            }
        }
        for output in graph.output.iter() {
            trace!("Model output: {output:?}");
        }
        for (name, t) in initializers.into_iter().sorted_by_key(|kv| kv.0) {
            let id = model.add_const(name, t)?;
            outlets_by_name.insert(name.to_string(), id);
        }
        let consts = model.nodes().len();
        for pbnode in graph.node.iter() {
            let name = if !pbnode.name.is_empty() {
                pbnode.name.to_string()
            } else if pbnode.output.len() > 0 && !pbnode.output[0].is_empty() {
                pbnode.output[0].to_owned()
            } else {
                format!("{}-{}", model.nodes().len(), pbnode.op_type)
            };
            trace!("Creating node {name}");
            let facts = pbnode
                .output
                .iter()
                .filter(|s| !s.is_empty())
                .map(|_| InferenceFact::default())
                .collect();
            trace!("  outputs {:?}", pbnode.output);
            let (op, closures) = match self.framework.op_register.0.get(&pbnode.op_type) {
                Some(builder) => (builder)(&ctx, pbnode).with_context(|| {
                    format!("Building node {} ({})", pbnode.name, pbnode.op_type)
                })?,
                None => (
                    tract_hir::ops::unimpl::UnimplementedOp::new(
                        pbnode.output.len(),
                        &*pbnode.op_type,
                        format!("{pbnode:?}"),
                    )
                    .into(),
                    vec![],
                ),
            };
            let id = model.add_node(name, op, facts)?;
            for (ix, output) in pbnode.output.iter().filter(|s| !s.is_empty()).enumerate() {
                outlets_by_name.insert(output.to_owned(), OutletId::new(id, ix));
                model.set_outlet_label(OutletId::new(id, ix), output.to_owned())?;
            }
            for closure in closures {
                trace!("Node {} closes on {}", model.nodes()[id], closure);
                closures_to_wire.push((id, closure))
            }
        }
        for (id, pbnode) in graph.node.iter().enumerate() {
            for (ix, input) in pbnode.input.iter().filter(|s| !s.is_empty()).enumerate() {
                if !outlets_by_name.contains_key(input) {
                    let id = model.add_source(input.clone(), InferenceFact::default())?;
                    unresolved_inputs.push(input.to_string());
                    outlets_by_name.insert(input.to_string(), id);
                }
                let outlet = outlets_by_name[input];
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
            let mut fact = InferenceFact::default();
            if self.framework.use_output_shapes {
                if let Some(f) = output.r#type.as_ref().and_then(|t| t.value.as_ref()) {
                    let pb::type_proto::Value::TensorType(f) = f;
                    fact = translate_inference_fact(&ctx, f, false)?
                };
            }
            if self.framework.ignore_output_types {
                fact = fact.without_datum_type();
            }
            let outlet = outlets_by_name[&*output.name];
            outputs.push(outlet);
            model.set_outlet_label(outlet, output.name.clone())?;
            model.set_outlet_fact(outlet, fact)?;
        }
        model.set_output_outlets(&outputs)?;
        for info in &graph.value_info {
            if let Some(TypeProto { value: Some(Value::TensorType(t)), .. }) = &info.r#type {
                if let Some(outlet) = outlets_by_name.get(&info.name) {
                    let mut pbfact = translate_inference_fact(&ctx, t, false)?;
                    // be conservative, these are likely to be TDim
                    if pbfact.datum_type() == Some(i64::datum_type()) {
                        pbfact = pbfact.without_datum_type();
                    }
                    model.set_outlet_fact(*outlet, pbfact)?;
                }
            }
        }
        let result = ParseResult { model, unresolved_inputs, outlets_by_name };
        Ok(result)
    }
}

type OpBuilder =
    fn(&ParsingContext, node: &pb::NodeProto) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)>;

#[derive(Clone, Default)]
pub struct OnnxOpRegister(pub HashMap<String, OpBuilder>);

impl OnnxOpRegister {
    pub fn insert(&mut self, s: &'static str, builder: OpBuilder) {
        self.0.insert(s.into(), builder);
    }
}

#[derive(Clone)]
pub struct Onnx {
    pub op_register: OnnxOpRegister,
    pub use_output_shapes: bool,
    pub ignore_output_types: bool,
    pub provider: Arc<dyn ModelDataResolver + Send + Sync>,
}

impl Default for Onnx {
    fn default() -> Self {
        Onnx {
            op_register: Default::default(),
            use_output_shapes: Default::default(),
            ignore_output_types: Default::default(),
            provider: Arc::new(data_resolver::MmapDataResolver),
        }
    }
}

impl Onnx {
    pub fn parse(&self, proto: &pb::ModelProto, path: Option<&str>) -> TractResult<ParseResult> {
        self.parse_with_template(proto, path, Default::default())
    }
    pub fn parse_with_template(
        &self,
        proto: &pb::ModelProto,
        model_dir: Option<&str>,
        template: InferenceModel,
    ) -> TractResult<ParseResult> {
        let onnx_operator_set_version = proto
            .opset_import
            .iter()
            .find(|import| import.domain.is_empty() || import.domain == "ai.onnx")
            .map(|op| op.version)
            .unwrap_or(0);
        let graph =
            proto.graph.as_ref().ok_or_else(|| anyhow!("model proto does not contain a graph"))?;
        debug!("ONNX operator set version: {onnx_operator_set_version:?}");
        if onnx_operator_set_version != 0 && !(9..19).contains(&onnx_operator_set_version) {
            warn!("ONNX operator for your model is {onnx_operator_set_version}, tract is only tested against \
                  operator set 9 to 18 (included). Your model may still work so this is not a hard fail.");
        }
        let ctx = ParsingContext {
            framework: self,
            model: proto,
            parent_graphs: vec![],
            onnx_operator_set_version,
            model_dir,
            template,
        };
        trace!("created ParsingContext");
        ctx.parse_graph(graph)
    }

    pub fn with_ignore_output_shapes(self, ignore: bool) -> Onnx {
        Self { use_output_shapes: !ignore, ..self }
    }

    pub fn with_ignore_output_types(self, ignore: bool) -> Onnx {
        Self { ignore_output_types: ignore, ..self }
    }

    pub fn determinize(model: &mut InferenceModel) -> TractResult<()> {
        use crate::ops::multinomial::Multinomial;
        for node in model.nodes_mut() {
            if let Some(op) = node.op_as_mut::<Box<dyn Expansion>>() {
                if let Some(op) = op.as_any_mut().downcast_mut::<Multinomial>() {
                    op.seed.get_or_insert(1.0);
                }
            }
        }
        Ok(())
    }
}

impl Framework<pb::ModelProto, InferenceModel> for Onnx {
    fn model_for_path(&self, p: impl AsRef<path::Path>) -> TractResult<InferenceModel> {
        let mut path = PathBuf::new();
        path.push(&p);
        let mut dir: Option<&str> = None;
        if let Some(dir_opt) = path.parent() {
            dir = dir_opt.to_str();
        }
        let proto = self.proto_model_for_path(p)?;
        let ParseResult { model, unresolved_inputs, .. } = self.parse(&proto, dir)?;
        if unresolved_inputs.len() > 0 {
            bail!("Could not resolve inputs at top-level: {:?}", unresolved_inputs)
        }
        Ok(model)
    }

    #[cfg(target_family = "wasm")]
    fn proto_model_for_path(&self, p: impl AsRef<path::Path>) -> TractResult<pb::ModelProto> {
        let p = p.as_ref();
        let mut file = fs::File::open(p).with_context(|| format!("Opening {p:?}"))?;
        Ok(self.proto_model_for_read(&mut file)?)
    }

    #[cfg(not(target_family = "wasm"))]
    fn proto_model_for_path(&self, p: impl AsRef<path::Path>) -> TractResult<pb::ModelProto> {
        let p = p.as_ref();
        let map = unsafe {
            memmap2::Mmap::map(&fs::File::open(p).with_context(|| format!("Opening {p:?}"))?)?
        };
        Ok(crate::pb::ModelProto::decode(&*map)?)
    }

    fn proto_model_for_read(&self, r: &mut dyn std::io::Read) -> TractResult<pb::ModelProto> {
        let mut v = vec![];
        r.read_to_end(&mut v)?;
        let b = bytes::Bytes::from(v);
        Ok(crate::pb::ModelProto::decode(b)?)
    }

    fn model_for_proto_model_with_model_template(
        &self,
        proto: &pb::ModelProto,
        template: InferenceModel,
    ) -> TractResult<InferenceModel> {
        let ParseResult { model, unresolved_inputs, .. } =
            self.parse_with_template(proto, None, template)?;
        if unresolved_inputs.len() > 0 {
            bail!("Could not resolve inputs at top-level: {:?}", unresolved_inputs)
        }
        Ok(model)
    }

    fn model_for_read(&self, r: &mut dyn std::io::Read) -> TractResult<InferenceModel> {
        let proto_model = self.proto_model_for_read(r).context("Reading proto model")?;
        self.model_for_proto_model(&proto_model).context("Translating proto model to model")
    }
}
