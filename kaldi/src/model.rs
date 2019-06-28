use tract_core::internal::*;

#[derive(Clone, Debug, Default)]
pub struct KaldiProtoModel {
    pub config_lines: ConfigLines,
    pub components: HashMap<String, Component>,
}

#[derive(Clone, Debug, Default)]
pub struct ConfigLines {
    pub input_name: String,
    pub input_dim: usize,
    pub component_nodes: HashMap<String, ComponentNode>,
    pub output_name: String,
    pub output_input: String,
}

#[derive(Clone, Debug, Default)]
pub struct ComponentNode {
    pub input: String,
    pub component: String,
}

#[derive(Clone, Debug, Default)]
pub struct Component {
    pub klass: String,
    pub attributes: HashMap<String, Arc<Tensor>>,
}

pub struct ParsingContext<'a> {
    pub proto_model: &'a KaldiProtoModel,
}

#[derive(Clone, Default)]
pub struct KaldiOpRegister(pub HashMap<String, fn(&ParsingContext, node: &str) -> TractResult<Box<InferenceOp>>>);

impl KaldiOpRegister {
    pub fn insert(&mut self, s: &'static str, builder: fn(&ParsingContext, node: &str) -> TractResult<Box<InferenceOp>>) {
        self.0.insert(s.into(), builder);
    }
}

#[derive(Clone, Default)]
pub struct Kaldi {
    pub op_register: KaldiOpRegister
}

impl Framework<KaldiProtoModel> for Kaldi {
    fn proto_model_for_read(&self, r: &mut std::io::Read) -> TractResult<KaldiProtoModel> {
        use crate::parser;
        let mut v = vec![];
        r.read_to_end(&mut v)?;
        parser::nnet3(&*v)
    }

    fn model_for_proto_model(&self, proto_model: &KaldiProtoModel) -> TractResult<InferenceModel> {
        let ctx = ParsingContext { proto_model };
        let mut model = InferenceModel::default();
        model.add_source(
            proto_model.config_lines.input_name.clone(),
            TensorFact::dt_shape(
                f32::datum_type(),
                shapefact!(_, (proto_model.config_lines.input_dim)),
            ),
        )?;
        for (name, node) in &proto_model.config_lines.component_nodes {
            let component = &proto_model.components[&node.component];
            let op = match self.op_register.0.get(&*component.klass) {
                Some(builder) => (builder)(&ctx, name)?,
                None => tract_core::ops::unimpl::UnimplementedOp::new(
                    component.klass.to_string(),
                    format!("{:?}", proto_model.config_lines.component_nodes.get(name)),
                )
                .into(),
            };
            model.add_node_default(name.to_string(), op)?;
        }
        for (name, node) in &proto_model.config_lines.component_nodes {
            let src = OutletId::new(model.node_by_name(&*node.input)?.id, 0);
            let dst = InletId::new(model.node_by_name(name)?.id, 0);
            model.add_edge(src, dst)?;
        }
        model.set_output_names(&[&*proto_model.config_lines.output_input])?;
        Ok(model)
    }
}
