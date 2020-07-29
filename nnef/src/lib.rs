pub mod ast;
pub mod model;
pub mod parser;
pub mod primitives;

pub use model::ProtoModel;
use tract_core::internal::*;

pub fn open_model<P: AsRef<std::path::Path>>(p: P) -> TractResult<ProtoModel> {
    use std::io::Read;
    let path = p.as_ref();
    let (text, _tensors) = if !path.exists() {
        bail!("File not found: {:?}", path)
    } else if path.is_dir() && path.join("graph.nnef").is_file() {
        (std::fs::read_to_string(path.join("graph.nnef"))?, ())
    } else if path.is_file()
        && path
            .file_name()
            .map(|s| [".tgz", ".tar.gz"].iter().any(|ext| s.to_string_lossy().ends_with(ext)))
            .unwrap_or(false)
    {
        let file = std::fs::File::open(path)?;
        let decomp = flate2::read::GzDecoder::new(file);
        let mut tar = tar::Archive::new(decomp);
        let mut text = None;
        for entry in tar.entries()? {
            let mut entry = entry?;
            if entry.path()?.file_name().map(|n| n == "graph.nnef").unwrap_or(false) {
                let mut t = String::new();
                entry.read_to_string(&mut t)?;
                text = Some(t)
            }
        }
        let text = text.ok_or_else(|| format!("Archive must contain graph.nnef at top level"))?;
        (text, ())
    } else {
        bail!("Model expected as a tar.gz archive of a directory")
    };
    let doc = parser::parse_document(&text)?;
    Ok(ProtoModel { doc })
}
