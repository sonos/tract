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
    ) -> TractResult<Option<(String, Arc<dyn Resource>)>> {
        if path.to_str() == Some(GRAPH_NNEF_FILENAME) {
            let mut text = String::new();
            reader.read_to_string(&mut text)?;
            let document = crate::ast::parse::parse_document(&text)?;
            Ok(Some((GRAPH_NNEF_FILENAME.to_string(), Arc::new(document))))
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
    ) -> TractResult<Option<(String, Arc<dyn Resource>)>> {
        if path.extension().map(|e| e == "dat").unwrap_or(false) {
            let tensor = crate::tensors::read_tensor(reader)
                .with_context(|| format!("Error while reading tensor {:?}", path))?;
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
    ) -> TractResult<Option<(String, Arc<dyn Resource>)>> {
        if path.to_str() == Some(GRAPH_QUANT_FILENAME) {
            let mut t = String::new();
            reader.read_to_string(&mut t)?;
            let quant = crate::ast::quant::parse_quantization(&t)?;
            let quant: HashMap<String, QuantFormat> =
                quant.into_iter().map(|(k, v)| (k.0, v)).collect();
            Ok(Some((GRAPH_QUANT_FILENAME.to_string(), Arc::new(quant))))
        } else {
            Ok(None)
        }
    }
}
