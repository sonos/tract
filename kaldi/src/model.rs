use tract_core::internal::*;

#[derive(Clone, Debug, Default)]
pub struct ConfigLines {
    pub input_name: String,
    pub input_dim: usize,
    pub component_nodes: HashMap<String, (String, String)>,
    pub output_name: String,
    pub output_input: String,
}

#[derive(Clone, Debug, Default)]
pub struct ProtoComponent {
    pub klass: String,
    pub attributes: HashMap<String, Tensor>,
}

#[derive(Clone, Debug, Default)]
pub struct KaldiProtoModel {
    pub config_lines: ConfigLines,
    pub components: HashMap<String, ProtoComponent>,
}

#[derive(Clone, Debug, Default)]
pub struct Kaldi {}

impl Framework<KaldiProtoModel> for Kaldi {
    fn proto_model_for_read(&self, r: &mut std::io::Read) -> TractResult<KaldiProtoModel> {
        let mut v = vec![];
        r.read_to_end(&mut v)?;
        crate::parser::nnet3(&*v)
    }

    fn model_for_proto_model(&self, proto: &KaldiProtoModel) -> TractResult<InferenceModel> {
        unimplemented!()
    }
}
