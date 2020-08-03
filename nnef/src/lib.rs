pub mod ast;
pub mod dump;
pub mod model;
pub mod parser;
pub mod primitives;
pub mod tensors;

pub use model::ProtoModel;
use tract_core::internal::*;
pub use tract_core::prelude;

pub fn open_model<P: AsRef<std::path::Path>>(p: P) -> TractResult<ProtoModel> {
    let path = p.as_ref();
    let mut text: Option<String> = None;
    let mut tensors: std::collections::HashMap<String, Arc<Tensor>> = Default::default();
    if !path.exists() {
        bail!("File not found: {:?}", path)
    } else if path.is_dir() && path.join("graph.nnef").is_file() {
        for entry in walkdir::WalkDir::new(path) {
            let entry = entry.map_err(|e| format!("Can not walk directory {:?}: {:?}", path, e))?;
            let path = entry.path();
            let mut stream = std::fs::File::open(path)?;
            process_stream(&path, &mut stream, &mut text, &mut tensors)?;
        }
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
            let path = entry.path()?.to_path_buf();
            process_stream(&path, &mut entry, &mut text, &mut tensors)?;
        }
    } else {
        bail!("Model expected as a tar.gz archive of a directory containing a file called `graph.nnef'")
    };
    let text = text.ok_or_else(|| format!("Model must contain graph.nnef at top level"))?;
    let doc = parser::parse_document(&text)?;
    Ok(ProtoModel { doc, tensors })
}

fn process_stream<R: std::io::Read>(
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
        let tensor = tensors::read_tensor(reader)?;
        tensors.insert(id.to_string(), tensor.into_arc_tensor());
    }
    Ok(())
}
