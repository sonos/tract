use tar::Builder;
use tract_core::tract_data::itertools::Itertools;

use crate::ast::quant::write_quant_format;
use crate::ast::{Document, Identifier, LazyReader, ProtoModel, QuantFormat};
use crate::resource::{LazyDat, LazyDatLoader};
use crate::{internal::*, nnef};
use std::io::Read;
#[cfg(target_family = "unix")]
use std::os::unix::prelude::OsStrExt;
use std::path::Path;
use std::str::FromStr;

pub fn stdlib() -> Vec<FragmentDef> {
    crate::ast::parse::parse_fragments(include_str!("../stdlib.nnef")).unwrap()
}

pub struct Nnef {
    pub stdlib: Vec<FragmentDef>,
    pub registries: Vec<Registry>,
    pub resource_loaders: Vec<Box<dyn ResourceLoader + 'static>>,
    pub allow_extended_identifier_syntax: bool,
}

impl Default for Nnef {
    fn default() -> Nnef {
        Nnef {
            stdlib: stdlib(),
            registries: vec![crate::ops::tract_nnef()],
            resource_loaders: vec![
                LazyDatLoader.into_boxed(),
                GraphNnefLoader.into_boxed(),
                DatLoader.into_boxed(),
                GraphQuantLoader.into_boxed(),
                TypedModelLoader::new(false).into_boxed(),
            ],
            allow_extended_identifier_syntax: false,
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

    pub fn enable_tract_core(&mut self) {
        self.registries.push(crate::ops::tract_core());
    }

    pub fn with_tract_core(mut self) -> Self {
        self.registries.push(crate::ops::tract_core());
        self
    }

    pub fn enable_tract_resource(&mut self) {
        self.registries.push(crate::ops::tract_resource());
    }

    pub fn with_tract_resource(mut self) -> Self {
        self.registries.push(crate::ops::tract_resource());
        self
    }

    pub fn allow_extended_identifier_syntax(&mut self, allow_extended_identifier_syntax: bool) {
        self.allow_extended_identifier_syntax = allow_extended_identifier_syntax;
    }

    pub fn translate(
        &self,
        proto_model: &ProtoModel,
        symbols: &SymbolTable,
    ) -> Result<TypedModel, (TypedModel, TractError)> {
        ModelBuilder::new(self, proto_model, symbols).into_typed_model()
    }

    pub fn write(&self, model: &TypedModel, w: impl std::io::Write) -> TractResult<()> {
        self.write_to_tar(model, w)?;
        Ok(())
    }

    pub fn write_to_tar<W: std::io::Write>(&self, model: &TypedModel, w: W) -> TractResult<W> {
        let mut ar = tar::Builder::new(w);
        self._write_to_tar(model, &mut ar, false)?;
        ar.into_inner().context("Finalizing tar")
    }

    pub fn write_to_tar_with_config<W: std::io::Write>(
        &self,
        model: &TypedModel,
        w: W,
        compress_nested_models: bool,
    ) -> TractResult<W> {
        let mut ar = tar::Builder::new(w);
        self._write_to_tar(model, &mut ar, compress_nested_models)?;
        ar.into_inner().context("Finalizing tar")
    }

    fn _write_to_tar<W: std::io::Write>(
        &self,
        model: &TypedModel,
        ar: &mut Builder<W>,
        compress_nested_models: bool,
    ) -> TractResult<()> {
        let proto_model =
            crate::ser::to_proto_model(self, model).context("Translating model to proto_model")?;

        let mut graph_data = vec![];
        crate::ast::dump::Dumper::new(self, &mut graph_data)
            .document(&proto_model.doc)
            .context("Serializing graph.nnef")?;
        let now =
            std::time::SystemTime::now().duration_since(std::time::SystemTime::UNIX_EPOCH).unwrap();

        let mut header = tar::Header::new_gnu();
        header.set_path("graph.nnef").context("Setting graph.nnef path")?;
        header.set_size(graph_data.len() as u64);
        header.set_mode(0o644);
        header.set_mtime(now.as_secs());
        header.set_cksum();
        ar.append(&header, &mut &*graph_data).context("Appending graph.nnef")?;

        if let Some(quantization) = proto_model.quantization {
            let mut quant_data = vec![];

            for (name, format) in quantization.into_iter() {
                write_quant_format(
                    &mut quant_data,
                    &name,
                    format,
                    self.allow_extended_identifier_syntax,
                )
                .context("Serializing graph.quant")?;
            }

            header.set_path("graph.quant").context("Setting graph.quant path")?;
            header.set_size(quant_data.len() as u64);
            header.set_mode(0o644);
            header.set_mtime(now.as_secs());
            header.set_cksum();
            ar.append(&header, &mut &*quant_data).context("Appending graph.quant")?;
        }

        for (label, t) in &proto_model.tensors {
            let mut label = label.0.to_string() + ".dat";
            if label.starts_with('/') {
                label.insert(0, '.');
            }
            let filename = std::path::Path::new(&label);
            let mut data = vec![];
            crate::tensors::write_tensor(&mut data, t)
                .with_context(|| format!("Serializing tensor {filename:?}: {t:?}"))?;

            let mut header = tar::Header::new_gnu();
            header.set_size(data.len() as u64);
            header.set_mode(0o644);
            header.set_mtime(now.as_secs());
            header.set_cksum();

            ar.append_data(&mut header, filename, &mut &*data)
                .with_context(|| format!("Appending tensor {filename:?}"))?;
        }

        for (label, resource) in proto_model.resources.iter() {
            if let Some(typed_model_resource) = resource.downcast_ref::<TypedModelResource>() {
                let mut submodel_data = vec![];
                let mut filename = std::path::PathBuf::from_str(label)?;
                let typed_model = &typed_model_resource.0;

                if compress_nested_models {
                    filename.set_extension("nnef.tgz");
                    let encoder = flate2::write::GzEncoder::new(
                        &mut submodel_data,
                        flate2::Compression::default(),
                    );
                    self.write(typed_model, encoder)?;
                } else {
                    filename.set_extension("nnef.tar");
                    self.write(typed_model, &mut submodel_data)?;
                }

                let mut header = tar::Header::new_gnu();
                header.set_size(submodel_data.len() as u64);
                header.set_mode(0o644);
                header.set_mtime(now.as_secs());
                header.set_cksum();

                ar.append_data(&mut header, filename, &mut &*submodel_data)
                    .with_context(|| format!("Appending submodel {label:?}"))?;
            }
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
        let proto_model = crate::ser::to_proto_model(self, model)?;
        std::fs::create_dir_all(path)?;
        let mut graph_nnef = std::fs::File::create(path.join("graph.nnef"))?;
        crate::ast::dump::Dumper::new(self, &mut graph_nnef).document(&proto_model.doc)?;

        if let Some(quantization) = proto_model.quantization {
            let mut graph_quant = std::fs::File::create(path.join("graph.quant"))?;
            for (name, format) in quantization.into_iter().sorted_by_key(|(x, _)| x.clone()) {
                write_quant_format(
                    &mut graph_quant,
                    &name,
                    format,
                    self.allow_extended_identifier_syntax,
                )?;
            }
        }

        for (label, t) in &proto_model.tensors {
            let label = label.0.to_string() + ".dat";
            let label = label.trim_start_matches('/');
            let parent = path.join(label).parent().unwrap().to_owned();
            std::fs::create_dir_all(&parent).with_context(|| format!("Creating dir {parent:?}"))?;
            let filename = path.join(label).to_owned();
            let mut file = std::fs::File::create(&filename)
                .with_context(|| format!("Creating file {filename:?}"))?;

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

        // `walkdir::new` will first yield the given path at depth 0, but we don't want to load this
        // entry here: only its descendants at depth >= 1.
        for entry in walkdir::WalkDir::new(path).min_depth(1) {
            let entry =
                entry.map_err(|e| format_err!("Can not walk directory {:?}: {:?}", path, e))?;
            // We don't want to load sub-directories themselves either.
            if entry.path().is_dir() {
                continue;
            }
            let subpath = entry
                .path()
                .components()
                .skip(path.components().count())
                .collect::<std::path::PathBuf>();
            let mut stream = std::fs::File::open(entry.path())?;
            read_stream(
                &subpath,
                Some(LazyReader::File(entry.path().to_owned())),
                &mut stream,
                &mut resources,
                self,
            )?;
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
            read_stream(&path, None, &mut entry, &mut resources, self)?;
        }
        proto_model_from_resources(resources)
    }

    fn model_for_proto_model_with_symbols(
        &self,
        proto: &ProtoModel,
        symbols: &SymbolTable,
    ) -> TractResult<TypedModel> {
        self.translate(proto, symbols).map_err(|e| e.1)
    }
}

fn proto_model_from_resources(
    resources: HashMap<String, Arc<dyn Resource>>,
) -> TractResult<ProtoModel> {
    // Iter resources IDs to detect submodels. Submodels are IDs with
    // - two path compoents (ex: XXX/file)
    // - a graph.nnef file as filename
    let sub_model_ids = resources
        .keys()
        .clone()
        .filter_map(|id| {
            let id_components = id.split('/').collect::<Vec<_>>();
            if (id_components.last() == Some(&crate::resource::GRAPH_NNEF_FILENAME))
                & (id_components.len() == 2)
            {
                id_components.first().map(|it| it.to_string())
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    // If there are submodels, we use the associated resources to create a TypedModel resource and add
    // it as a new resource.
    let new_resources = if sub_model_ids.len() > 0 {
        sub_model_ids.into_iter().try_fold(resources, |r, it| -> TractResult<HashMap<_, _>> {
            let (submodel_resources, mut resources): (HashMap<String, Arc<dyn Resource>>, _) =
                r.into_iter().partition(|(k, _v)| k.starts_with(&it));
            let submodel_resources = submodel_resources
                .into_iter()
                .map(|(k, v)| (k.split('/').last().unwrap().to_string(), v))
                .collect::<HashMap<String, Arc<dyn Resource>>>();
            let typed_model = nnef()
                .model_for_proto_model(&proto_model_from_resources(submodel_resources).unwrap())?;
            resources.insert(it, Arc::new(TypedModelResource(typed_model)));
            Ok(resources)
        })?
    } else {
        resources
    };

    let mut resources = HashMap::default();
    let mut tensors = HashMap::default();
    let mut lazy_tensors = HashMap::default();
    let mut doc: Option<Arc<Document>> = None;
    let mut quantization = None;
    for (k, res) in new_resources {
        if let Ok(t) = res.clone().downcast_arc::<Tensor>() {
            tensors.insert(Identifier(k), t);
        } else if let Ok(t) = res.clone().downcast_arc::<LazyDat>() {
            lazy_tensors.insert(Identifier(k), t);
        } else if k == crate::resource::GRAPH_NNEF_FILENAME {
            doc = Some(
                res.downcast_arc::<Document>()
                    .map_err(|_| anyhow!("graph.nnef must be a Document"))?,
            );
        } else if k == crate::resource::GRAPH_QUANT_FILENAME {
            let map = res
                .downcast_arc::<HashMap<String, QuantFormat>>()
                .map_err(|_| anyhow!("graph.quant must be quantization information"))?;
            quantization =
                Some(map.iter().map(|(k, v)| (Identifier::from(&**k), v.clone())).collect())
        } else {
            resources.insert(k, res);
        }
    }

    let Some(doc) = doc else { bail!("Could not find graph.nnef") };
    let doc = Arc::try_unwrap(doc).unwrap();

    let proto = ProtoModel { doc, tensors, lazy_tensors, quantization, resources };
    proto.validate()?;
    Ok(proto)
}

fn read_stream(
    path: &Path,
    lazy_data_provider: Option<LazyReader>,
    reader: &mut impl std::io::Read,
    resources: &mut HashMap<String, Arc<dyn Resource>>,
    framework: &Nnef,
) -> TractResult<()> {
    // ignore path with any component starting with "." (because OSX's tar is weird)
    #[cfg(target_family = "unix")]
    if path.components().any(|name| name.as_os_str().as_bytes().first() == Some(&b'.')) {
        return Ok(());
    }
    let mut last_loader_name;
    for loader in framework.resource_loaders.iter() {
        last_loader_name = Some(loader.name());
        let loaded = loader
            .try_load(path, lazy_data_provider.clone(), reader, framework)
            .with_context(|| {
                anyhow!("Error while loading resource by {:?} at path {:?}", loader.name(), path)
            })?;
        if let Some((id, resource)) = loaded {
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
