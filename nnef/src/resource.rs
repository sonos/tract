use std::path::Path;

use crate::ast::{Document, QuantFormat};
use crate::internal::*;
use tract_core::downcast_rs::{impl_downcast, DowncastSync};

pub const GRAPH_NNEF_FILENAME: &str = "graph.nnef";
pub const GRAPH_QUANT_FILENAME: &str = "graph.quant";

pub fn resource_path_to_id(path: impl AsRef<Path>) -> TractResult<String> {
    let mut path = path.as_ref().to_path_buf();
    path.set_extension("");
    path.to_str()
        .ok_or_else(|| format_err!("Badly encoded filename for path: {:?}", path))
        .map(|s| s.to_string())
}

pub trait Resource: DowncastSync + std::fmt::Debug + Send + Sync {
    /// Get value for a given key.
    fn get(&self, _key: &str) -> TractResult<Value> {
        bail!("No key access supported by this resource");
    }
}

impl_downcast!(sync Resource);

pub trait ResourceLoader: Send + Sync {
    /// Name of the resource loader.
    fn name(&self) -> Cow<str>;
    /// Try to load a resource give a path and its corresponding reader.
    /// None is returned if the path is not accepted by this loader.
    fn try_load(
        &self,
        path: &Path,
        reader: &mut dyn std::io::Read,
        framework: &Nnef,
    ) -> TractResult<Option<(String, Arc<dyn Resource>)>>;

    fn into_boxed(self) -> Box<dyn ResourceLoader>
    where
        Self: Sized + 'static,
    {
        Box::new(self)
    }
}

impl Resource for Document {}

#[derive(Clone, Debug, Copy, PartialEq, Eq, Hash)]
pub struct GraphNnefLoader;

impl ResourceLoader for GraphNnefLoader {
    fn name(&self) -> Cow<str> {
        "GraphNnefLoader".into()
    }

    fn try_load(
        &self,
        path: &Path,
        reader: &mut dyn std::io::Read,
        _framework: &Nnef,
    ) -> TractResult<Option<(String, Arc<dyn Resource>)>> {
        if path.ends_with(GRAPH_NNEF_FILENAME) {
            let mut text = String::new();
            reader.read_to_string(&mut text)?;
            let document = crate::ast::parse::parse_document(&text)?;
            Ok(Some((path.to_str().unwrap().to_string(), Arc::new(document))))
        } else {
            Ok(None)
        }
    }
}

impl Resource for Tensor {}

#[derive(Clone, Debug, Copy, PartialEq, Eq, Hash)]
pub struct DatLoader;

impl ResourceLoader for DatLoader {
    fn name(&self) -> Cow<str> {
        "DatLoader".into()
    }

    fn try_load(
        &self,
        path: &Path,
        reader: &mut dyn std::io::Read,
        _framework: &Nnef,
    ) -> TractResult<Option<(String, Arc<dyn Resource>)>> {
        if path.extension().map(|e| e == "dat").unwrap_or(false) {
            let tensor = crate::tensors::read_tensor(reader)
                .with_context(|| format!("Error while reading tensor {path:?}"))?;
            Ok(Some((resource_path_to_id(path)?, Arc::new(tensor))))
        } else {
            Ok(None)
        }
    }
}

impl Resource for HashMap<String, QuantFormat> {}

#[derive(Clone, Debug, Copy, PartialEq, Eq, Hash)]
pub struct GraphQuantLoader;

impl ResourceLoader for GraphQuantLoader {
    fn name(&self) -> Cow<str> {
        "GraphQuantLoader".into()
    }

    fn try_load(
        &self,
        path: &Path,
        reader: &mut dyn std::io::Read,
        _framework: &Nnef,
    ) -> TractResult<Option<(String, Arc<dyn Resource>)>> {
        if path.ends_with(GRAPH_QUANT_FILENAME) {
            let mut text = String::new();
            reader.read_to_string(&mut text)?;
            let quant = crate::ast::quant::parse_quantization(&text)?;
            let quant: HashMap<String, QuantFormat> =
                quant.into_iter().map(|(k, v)| (k.0, v)).collect();
            Ok(Some((path.to_str().unwrap().to_string(), Arc::new(quant))))
        } else {
            Ok(None)
        }
    }
}

pub struct TypedModelLoader {
    pub optimized_model: bool,
}

impl TypedModelLoader {
    pub fn new(optimized_model: bool) -> Self {
        Self { optimized_model }
    }
}

impl ResourceLoader for TypedModelLoader {
    fn name(&self) -> Cow<str> {
        "TypedModelLoader".into()
    }

    fn try_load(
        &self,
        path: &Path,
        reader: &mut dyn std::io::Read,
        framework: &Nnef,
    ) -> TractResult<Option<(String, Arc<dyn Resource>)>> {
        const NNEF_TGZ: &str = ".nnef.tgz";
        const NNEF_TAR: &str = ".nnef.tar";
        let path_str = path.to_str().unwrap_or("");
        if path_str.ends_with(NNEF_TGZ) || path_str.ends_with(NNEF_TAR) {
            let model = if self.optimized_model {
                framework.model_for_read(reader)?.into_optimized()?
            } else {
                framework.model_for_read(reader)?
            };

            let label = if path_str.ends_with(NNEF_TGZ) {
                path
                    .to_str()
                    .ok_or_else(|| anyhow!("invalid model resource path"))?
                    .trim_end_matches(NNEF_TGZ)
            } else {
                path
                    .to_str()
                    .ok_or_else(|| anyhow!("invalid model resource path"))?
                    .trim_end_matches(NNEF_TAR)
            };
            Ok(Some((resource_path_to_id(label)?, Arc::new(TypedModelResource(model)))))
        } else {
            Ok(None)
        }
    }
}

#[derive(Debug, Clone)]
pub struct TypedModelResource(pub TypedModel);

impl Resource for TypedModelResource {}
