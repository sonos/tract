use crate::ast::ProtoModel;
use crate::internal::*;

pub fn stdlib() -> Vec<FragmentDef> {
    crate::ast::parse::parse_fragments(include_str!("../stdlib.nnef")).unwrap()
}

pub struct Nnef {
    pub stdlib: Vec<FragmentDef>,
    pub registries: Vec<Registry>,
}

impl Nnef {
    pub fn new() -> Nnef {
        Nnef { stdlib: stdlib(), registries: vec![crate::ops::tract_nnef()] }
    }

    pub fn with_registry(mut self, registry: Registry) -> Nnef {
        self.registries.push(registry);
        self
    }

    pub fn translate(
        &self,
        proto_model: &ProtoModel,
    ) -> Result<TypedModel, (TypedModel, TractError)> {
        ModelBuilder::new(self, proto_model).into_typed_model()
    }

    pub fn write(&self, model: &TypedModel, w: impl std::io::Write) -> TractResult<()> {
        let proto_model = crate::ser::to_proto_model(&self, model)?;
        let comp = flate2::write::GzEncoder::new(w, flate2::Compression::default());
        let mut ar = tar::Builder::new(comp);
        let mut graph_data = vec![];
        crate::ast::dump::Dumper::new(&mut graph_data).document(&proto_model.doc)?;
        let now =
            std::time::SystemTime::now().duration_since(std::time::SystemTime::UNIX_EPOCH).unwrap();
        let mut header = tar::Header::new_gnu();
        header.set_path("graph.nnef")?;
        header.set_size(graph_data.len() as u64);
        header.set_mode(0o644);
        header.set_mtime(now.as_secs());
        header.set_cksum();
        ar.append(&header, &mut &*graph_data)?;

        for (label, t) in &proto_model.tensors {
            let filename = std::path::Path::new(label).to_path_buf().with_extension("dat");
            let mut data = vec![];
            crate::tensors::write_tensor(&mut data, t)?;

            let mut header = tar::Header::new_gnu();
            header.set_path(filename)?;
            header.set_size(data.len() as u64);
            header.set_mode(0o644);
            header.set_mtime(now.as_secs());
            header.set_cksum();

            ar.append(&header, &mut &*data)?;
        }
        Ok(())
    }

    pub fn write_to_dir(
        &self,
        model: &TypedModel,
        path: impl AsRef<std::path::Path>,
    ) -> TractResult<()> {
        let path = path.as_ref();
        if path.exists() {
            bail!("{:?} already exists. Won't overwrite.", path);
        }
        let proto_model = crate::ser::to_proto_model(&self, model)?;
        std::fs::create_dir_all(path)?;
        let mut graph_nnef = std::fs::File::create(path.join("graph.nnef"))?;
        crate::ast::dump::Dumper::new(&mut graph_nnef).document(&proto_model.doc)?;
        for (label, t) in &proto_model.tensors {
            let label = std::path::Path::new(&label);
            std::fs::create_dir_all(path.join(label).parent().unwrap())?;
            let filename = path.join(label).with_extension("dat");
            let mut file = std::fs::File::create(filename)?;
            crate::tensors::write_tensor(&mut file, t)?;
        }
        Ok(())
    }

    pub fn write_to_tgz(
        &self,
        model: &TypedModel,
        path: impl AsRef<std::path::Path>,
    ) -> TractResult<()> {
        let path = path.as_ref();
        if path.exists() {
            bail!("{:?} already exists. Won't overwrite.", path);
        }
        let file = std::fs::File::create(path)?;
        self.write(model, file)
    }
}

impl tract_core::prelude::Framework<ProtoModel, TypedModel> for Nnef {
    fn proto_model_for_read(&self, reader: &mut dyn std::io::Read) -> TractResult<ProtoModel> {
        let mut text: Option<String> = None;
        let mut tensors: std::collections::HashMap<String, Arc<Tensor>> = Default::default();
        let decomp = flate2::read::GzDecoder::new(reader);
        let mut tar = tar::Archive::new(decomp);
        for entry in tar.entries()? {
            let mut entry = entry?;
            let path = entry.path()?.to_path_buf();
            read_stream(&path, &mut entry, &mut text, &mut tensors)?;
        }
        let text = text.ok_or_else(|| format!("Model must contain graph.nnef at top level"))?;
        let doc = crate::ast::parse::parse_document(&text)?;
        Ok(ProtoModel { doc, tensors })
    }

    fn model_for_proto_model(&self, proto: &ProtoModel) -> TractResult<TypedModel> {
        self.translate(proto).map_err(|e| e.1)
    }
}

fn read_stream<R: std::io::Read>(
    path: &std::path::Path,
    reader: &mut R,
    text: &mut Option<String>,
    tensors: &mut HashMap<String, Arc<Tensor>>,
) -> TractResult<()> {
    if path.file_name().map(|n| n == "graph.nnef").unwrap_or(false) {
        let mut t = String::new();
        reader.read_to_string(&mut t)?;
        *text = Some(t);
    } else if path.extension().map(|e| e == "dat").unwrap_or(false) {
        let mut path = path.to_path_buf();
        path.set_extension("");
        let id = path
            .to_str()
            .ok_or_else(|| format!("Badly encoded filename for tensor: {:?}", path))?;
        let tensor = crate::tensors::read_tensor(reader)?;
        tensors.insert(id.to_string(), tensor.into_arc_tensor());
    }
    Ok(())
}
