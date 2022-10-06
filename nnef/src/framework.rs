use tract_core::tract_data::itertools::Itertools;

use crate::ast::quant::write_quant_format;
use crate::ast::{Document, ProtoModel, QuantFormat};
use crate::internal::*;
use std::io::Read;
#[cfg(target_family = "unix")]
use std::os::unix::prelude::OsStrExt;
use std::path::Path;

pub fn stdlib() -> Vec<FragmentDef> {
    crate::ast::parse::parse_fragments(include_str!("../stdlib.nnef")).unwrap()
}

pub struct Nnef {
    pub stdlib: Vec<FragmentDef>,
    pub registries: Vec<Registry>,
    pub resource_loaders: Vec<Box<dyn ResourceLoader + 'static>>,
}

impl Default for Nnef {
    fn default() -> Nnef {
        Nnef {
            stdlib: stdlib(),
            registries: vec![crate::ops::tract_nnef()],
            resource_loaders: vec![
                GraphNnefLoader.into_boxed(),
                DatLoader.into_boxed(),
                QuantFormatLoader.into_boxed(),
            ],
        }
    }
}

impl Nnef {
    pub fn with_registry(mut self, registry: Registry) -> Nnef {
        self.registries.push(registry);
        self
    }

    pub fn with_resource_loader(mut self, loader: impl ResourceLoader + 'static) -> Nnef {
        self.resource_loaders.push(Box::new(loader));
        self
    }

    pub fn with_tract_core(mut self) -> Self {
        self.registries.push(crate::ops::tract_core());
        self
    }

    pub fn with_tract_resource(mut self) -> Self {
        self.registries.push(crate::ops::tract_resource());
        self
    }

    pub fn translate(
        &self,
        proto_model: &ProtoModel,
    ) -> Result<TypedModel, (TypedModel, TractError)> {
        ModelBuilder::new(self, proto_model).into_typed_model()
    }

    pub fn write(&self, model: &TypedModel, w: impl std::io::Write) -> TractResult<()> {
        self.write_to_tar(model, w)?;
        Ok(())
    }

    pub fn write_to_tar<W: std::io::Write>(&self, model: &TypedModel, w: W) -> TractResult<W> {
        let proto_model =
            crate::ser::to_proto_model(self, model).context("Translating model to proto_model")?;
        let mut ar = tar::Builder::new(w);
        let mut graph_data = vec![];
        crate::ast::dump::Dumper::new(&mut graph_data)
            .document(&proto_model.doc)
            .context("Serializing graph.nnef")?;
        let now =
            std::time::SystemTime::now().duration_since(std::time::SystemTime::UNIX_EPOCH).unwrap();
        let mut header = tar::Header::new_gnu();
        header.set_path("graph.nnef")?;
        header.set_size(graph_data.len() as u64);
        header.set_mode(0o644);
        header.set_mtime(now.as_secs());
        header.set_cksum();
        ar.append(&header, &mut &*graph_data)?;

        if let Some(quantization) = proto_model.quantization {
            let mut quant_data = vec![];

            for (name, format) in quantization.into_iter() {
                write_quant_format(&mut quant_data, name, format)
                    .context("Serializing graph.quant")?;
            }

            header.set_path("graph.quant")?;
            header.set_size(quant_data.len() as u64);
            header.set_mode(0o644);
            header.set_mtime(now.as_secs());
            header.set_cksum();
            ar.append(&header, &mut &*quant_data)?;
        }

        for (label, t) in &proto_model.tensors {
            let label = label.to_string() + ".dat";
            let filename = std::path::Path::new(&label);
            let mut data = vec![];
            crate::tensors::write_tensor(&mut data, t)
                .with_context(|| format!("Serializing tensor {:?}: {:?}", filename, t))?;

            let mut header = tar::Header::new_gnu();
            header.set_size(data.len() as u64);
            header.set_mode(0o644);
            header.set_mtime(now.as_secs());
            header.set_cksum();

            ar.append_data(&mut header, filename, &mut &*data)?;
        }
        Ok(ar.into_inner()?)
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
        let proto_model = crate::ser::to_proto_model(self, model)?;
        std::fs::create_dir_all(path)?;
        let mut graph_nnef = std::fs::File::create(path.join("graph.nnef"))?;
        crate::ast::dump::Dumper::new(&mut graph_nnef).document(&proto_model.doc)?;

        if let Some(quantization) = proto_model.quantization {
            let mut graph_quant = std::fs::File::create(path.join("graph.quant"))?;
            for (name, format) in quantization.into_iter().sorted_by_key(|(x, _)| x.clone()) {
                write_quant_format(&mut graph_quant, name, format)?;
            }
        }

        for (label, t) in &proto_model.tensors {
            let label = label.to_string() + ".dat";
            std::fs::create_dir_all(path.join(&label).parent().unwrap())?;
            let filename = path.join(label);
            let mut file = std::fs::File::create(filename)?;
            crate::tensors::write_tensor(&mut file, t)?;
        }
        Ok(())
    }
}

impl tract_core::prelude::Framework<ProtoModel, TypedModel> for Nnef {
    fn model_for_path(&self, p: impl AsRef<Path>) -> TractResult<TypedModel> {
        let proto = self.proto_model_for_path(p)?;
        self.model_for_proto_model(&proto)
    }

    fn proto_model_for_path(&self, path: impl AsRef<Path>) -> TractResult<ProtoModel> {
        let path = path.as_ref();
        if path.is_file() {
            let mut f = std::fs::File::open(path)?;
            return self.proto_model_for_read(&mut f);
        }

        let mut resources: HashMap<String, Arc<dyn Resource>> = Default::default();
        for entry in walkdir::WalkDir::new(path) {
            let entry =
                entry.map_err(|e| format_err!("Can not walk directory {:?}: {:?}", path, e))?;
            let subpath = entry
                .path()
                .components()
                .skip(path.components().count())
                .collect::<std::path::PathBuf>();
            let mut stream = std::fs::File::open(entry.path())?;
            read_stream(&self.resource_loaders, &subpath, &mut stream, &mut resources)?;
        }
        proto_model_from_resources(resources)
    }

    fn proto_model_for_read(&self, reader: &mut dyn std::io::Read) -> TractResult<ProtoModel> {
        let mut resources: HashMap<String, Arc<dyn Resource>> = Default::default();

        let mut buffer = vec![0u8; 2];
        reader.read_exact(&mut buffer)?;
        let header = std::io::Cursor::new(buffer.clone());
        let stream = header.chain(reader);
        let mut tar = if buffer == [0x1f, 0x8b] {
            #[cfg(feature = "flate2")]
            {
                let f = flate2::read::GzDecoder::new(stream);
                tar::Archive::new(Box::new(f) as Box<dyn Read>)
            }
            #[cfg(not(feature = "flate2"))]
            bail!("Cannot read gzip file without flate2 enabled.");
        } else {
            tar::Archive::new(Box::new(stream) as Box<dyn Read>)
        };
        for entry in tar.entries()? {
            let mut entry = entry?;
            let path = entry.path()?.to_path_buf();
            read_stream(&self.resource_loaders, &path, &mut entry, &mut resources)?;
        }
        proto_model_from_resources(resources)
    }

    fn model_for_proto_model(&self, proto: &ProtoModel) -> TractResult<TypedModel> {
        self.translate(proto).map_err(|e| e.1)
    }
}

fn proto_model_from_resources(
    mut resources: HashMap<String, Arc<dyn Resource>>,
) -> TractResult<ProtoModel> {
    // NNEF document extraction
    let doc = resources
        .remove(crate::resource::GRAPH_NNEF_FILENAME)
        .with_context(|| {
            anyhow!("Resource {} was not found in the model", crate::resource::GRAPH_NNEF_FILENAME)
        })?
        .downcast_arc::<Document>()
        .map_err(|_| anyhow!("Error while downcasting NNEF document resource"))?;

    let doc = Arc::try_unwrap(doc)
        .map_err(|_| anyhow!("Error while extracting NNEF Document from shared reference. Only one reference to the document is expected"))?;

    // Collect all resources that can be downcastable to Arc<Tensor>.
    let tensors: HashMap<_, _> = resources
        .iter()
        .filter_map(|(key, resource)| {
            Arc::clone(&resource).downcast_arc::<Tensor>().ok().map(|r| (key.to_string(), r))
        })
        .collect();
    // Iterate over tensors keys to remove them from the global resources hash map.
    tensors.keys().for_each(|k| {
        resources.remove(k);
    });

    // Quantization format resources extraction if present.
    let quantization = resources.remove(crate::resource::GRAPH_QUANT_FILENAME)
        .map(|q_r| {
            q_r
                .downcast_arc::<HashMap<String, QuantFormat>>()
                .map_err(|_| anyhow!("Error while downcasting quantization format resource"))
        })
        .transpose()?
        .map(|quant| {
            Arc::try_unwrap(quant)
                .map_err(|_| anyhow!("Error while extracting quantization format resource from shared reference. Only one reference to it is expected"))
        })
        .transpose()?;

    let proto = ProtoModel { doc, tensors, quantization, resources };
    proto.validate()?;
    Ok(proto)
}

fn read_stream<R: std::io::Read>(
    resource_loaders: &[Box<dyn ResourceLoader>],
    path: &Path,
    reader: &mut R,
    resources: &mut HashMap<String, Arc<dyn Resource>>,
) -> TractResult<()> {
    // ignore path with any component starting with "." (because OSX's tar is weird)
    #[cfg(target_family = "unix")]
    if path.components().any(|name| name.as_os_str().as_bytes().first() == Some(&b'.')) {
        return Ok(());
    }
    let mut last_loader_name;
    for loader in resource_loaders {
        last_loader_name = Some(loader.name());
        if let Some((id, resource)) = loader.try_load(path, reader)? {
            ensure!(
                !resources.contains_key(&id),
                "Loader {:?} succeeded to load {:?} which has been already loaded by {:?}",
                loader.name(),
                id,
                last_loader_name
            );
            resources.insert(id, resource);
            break;
        }
    }
    Ok(())
}
