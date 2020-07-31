pub mod ast;
pub mod model;
pub mod parser;
pub mod primitives;
pub mod tensors;

pub use model::ProtoModel;
use tract_core::internal::*;

pub fn open_model<P: AsRef<std::path::Path>>(p: P) -> TractResult<ProtoModel> {
    use std::io::Read;
    let path = p.as_ref();
    let mut text: Option<String> = None;
    let mut tensors: std::collections::HashMap<String, Arc<Tensor>> = Default::default();
    if !path.exists() {
        bail!("File not found: {:?}", path)
    } else if path.is_dir() && path.join("graph.nnef").is_file() {
        text = Some(std::fs::read_to_string(path.join("graph.nnef"))?);
    } else if path.is_file()
        && path
            .file_name()
            .map(|s| [".tgz", ".tar.gz"].iter().any(|ext| s.to_string_lossy().ends_with(ext)))
            .unwrap_or(false)
    {
        let file = std::fs::File::open(path)?;
        let decomp = flate2::read::GzDecoder::new(file);
        let mut tar = tar::Archive::new(decomp);
        for entry in tar.entries()? {
            let mut entry = entry?;
            if entry.path()?.file_name().map(|n| n == "graph.nnef").unwrap_or(false) {
                let mut t = String::new();
                entry.read_to_string(&mut t)?;
                text = Some(t);
            } else if entry.path()?.extension().map(|e| e == "dat").unwrap_or(false) {
                let mut path = entry.path()?.to_path_buf();
                path.set_extension("");
                let id = path
                    .to_str()
                    .ok_or_else(|| format!("Badly encoded filename for tensor: {:?}", path))?;
                let tensor = tensors::read_tensor(&mut entry)?;
                tensors.insert(id.to_string(), tensor.into_arc_tensor());
            }
        }
    } else {
        bail!("Model expected as a tar.gz archive of a directory")
    };
    let text = text.ok_or_else(|| format!("Model must contain graph.nnef at top level"))?;
    let doc = parser::parse_document(&text)?;
    Ok(ProtoModel { doc, tensors })
}
