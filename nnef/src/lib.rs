pub mod ast;
pub mod parser;

use tract_core::internal::*;

pub struct ProtoModel {
    doc: ast::Document,
}

pub fn open_model<P: AsRef<std::path::Path>>(p: P) -> TractResult<ProtoModel> {
    use std::io::Read;
    let path = p.as_ref();
    if !path.exists() {
        bail!("File not found: {:?}", path)
    } else if path.is_dir() && path.join("graph.nnef").is_file() {
        let nnef = std::fs::read_to_string(path.join("graph.nnef"))?;
        let doc = parser::parse_document(&nnef)?;
        Ok(ProtoModel { doc })
    } else if path.is_file()
        && path
            .file_name()
            .map(|s| [".tgz", ".tar.gz"].iter().any(|ext| s.to_string_lossy().ends_with(ext)))
            .unwrap_or(false)
    {
        let file = std::fs::File::open(path)?;
        let decomp = flate2::read::GzDecoder::new(file);
        let mut tar = tar::Archive::new(decomp);
        let mut doc = None;
        for entry in tar.entries()? {
            let mut entry = entry?;
            if entry.path()?.file_name().map(|n| n == "graph.nnef").unwrap_or(false) {
                let mut text = String::new();
                entry.read_to_string(&mut text)?;
                doc = Some(parser::parse_document(&text)?);
            }
        }
        let doc: ast::Document = if let Some(doc) = doc { doc } else {
            bail!("Archive must contain graph.nnef at top level")
        };
        Ok(ProtoModel { doc })
    } else {
        bail!("Model expected as a tar.gz archive of a directory")
    }
}
