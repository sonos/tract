//! The `.safetensors` format: a little-endian header length, a JSON header naming each tensor's
//! dtype, shape and byte range, then the tensors' bytes.

use std::collections::HashMap;
use std::fmt;

use crate::internal::*;

/// One tensor of a safetensors file: its datum type, shape and byte range in the data section.
#[derive(Clone, Debug)]
pub struct Entry {
    pub name: String,
    pub datum_type: DatumType,
    pub shape: Vec<usize>,
    pub begin: usize,
}

fn datum_type(dtype: &str) -> TractResult<DatumType> {
    Ok(match dtype {
        "F64" => DatumType::F64,
        "F32" => DatumType::F32,
        "F16" => DatumType::F16,
        "I64" => DatumType::I64,
        "I32" => DatumType::I32,
        "I16" => DatumType::I16,
        "I8" => DatumType::I8,
        "U64" => DatumType::U64,
        "U32" => DatumType::U32,
        "U16" => DatumType::U16,
        "U8" => DatumType::U8,
        _ => bail!("Unsupported safetensors dtype {dtype}"),
    })
}

/// The length of the header that follows the first eight bytes.
pub fn header_len(prefix: [u8; 8]) -> TractResult<usize> {
    let len = u64::from_le_bytes(prefix) as usize;
    ensure!(len <= 100 << 20, "safetensors header of {len} bytes");
    Ok(len)
}

/// The tensors a header lists, in file order.
pub fn entries(header: &[u8]) -> TractResult<Vec<Entry>> {
    let header: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(header)?;
    let mut entries = vec![];
    for (name, info) in header {
        if name == "__metadata__" {
            continue;
        }
        let datum_type = datum_type(info["dtype"].as_str().context("dtype")?)?;
        let shape: Vec<usize> = serde_json::from_value(info["shape"].clone())?;
        let [begin, end]: [usize; 2] = serde_json::from_value(info["data_offsets"].clone())?;
        ensure!(
            end - begin == shape.iter().product::<usize>() * datum_type.size_of(),
            "safetensors tensor {name}: {} bytes for {datum_type:?} {shape:?}",
            end - begin
        );
        entries.push(Entry { name, datum_type, shape, begin });
    }
    entries.sort_by_key(|e| e.begin);
    Ok(entries)
}

/// Bytes read at an offset: a file, or whatever else holds a safetensors file.
pub trait ReadAt: Send + Sync {
    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> std::io::Result<()>;
}

#[cfg(unix)]
impl ReadAt for std::fs::File {
    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
        std::os::unix::fs::FileExt::read_exact_at(self, buf, offset)
    }
}

#[cfg(windows)]
impl ReadAt for std::fs::File {
    fn read_exact_at(&self, mut buf: &mut [u8], mut offset: u64) -> std::io::Result<()> {
        while !buf.is_empty() {
            let n = std::os::windows::fs::FileExt::seek_read(self, buf, offset)?;
            if n == 0 {
                return Err(std::io::ErrorKind::UnexpectedEof.into());
            }
            buf = &mut buf[n..];
            offset += n as u64;
        }
        Ok(())
    }
}

/// The weights of a safetensors file, each read at its offset when first asked for: link it to
/// a model's named weights with [`LinkWeights`](tract_core::transform::LinkWeights). Only the
/// header is read up front.
pub struct SafetensorsSource {
    reader: Box<dyn ReadAt>,
    data_start: u64,
    tensors: HashMap<String, Entry>,
}

impl SafetensorsSource {
    pub fn new(reader: impl ReadAt + 'static) -> TractResult<SafetensorsSource> {
        let mut prefix = [0u8; 8];
        reader.read_exact_at(&mut prefix, 0)?;
        let len = header_len(prefix)?;
        let mut header = vec![0u8; len];
        reader.read_exact_at(&mut header, 8)?;
        let tensors = entries(&header)?.into_iter().map(|e| (e.name.clone(), e)).collect();
        Ok(SafetensorsSource { reader: Box::new(reader), data_start: 8 + len as u64, tensors })
    }

    #[cfg(any(unix, windows))]
    pub fn open(path: impl AsRef<std::path::Path>) -> TractResult<SafetensorsSource> {
        let path = path.as_ref();
        let file = std::fs::File::open(path).with_context(|| format!("Opening {path:?}"))?;
        Self::new(file).with_context(|| format!("Reading the header of {path:?}"))
    }
}

impl fmt::Debug for SafetensorsSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SafetensorsSource({} tensors)", self.tensors.len())
    }
}

impl WeightSource for SafetensorsSource {
    fn load(&self, name: &str, dt: DatumType, shape: &[usize]) -> TractResult<Tensor> {
        let entry = self.tensors.get(name).with_context(|| format!("No tensor {name:?}"))?;
        ensure!(
            entry.datum_type == dt && entry.shape == shape,
            "Tensor {name:?} is {:?} {:?}, expected {dt:?} {shape:?}",
            entry.datum_type,
            entry.shape
        );
        let mut tensor = Tensor::zero_dt(dt, shape)?;
        self.reader.read_exact_at(tensor.as_bytes_mut(), self.data_start + entry.begin as u64)?;
        Ok(tensor)
    }

    fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::ProtoModel;
    use crate::ast::parse::parse_document;
    use tract_core::ops::konst::Const;
    use tract_core::transform::{LinkWeights, ModelTransform};

    fn safetensors(dir: &std::path::Path, tensors: &[(&str, &str, &Tensor)]) -> TractResult<()> {
        let mut header = serde_json::Map::new();
        let mut data = vec![];
        for (name, dtype, t) in tensors {
            let begin = data.len();
            data.extend_from_slice(t.as_bytes());
            header.insert(
                name.to_string(),
                serde_json::json!({ "dtype": dtype, "shape": t.shape(), "data_offsets": [begin, data.len()] }),
            );
        }
        let header = serde_json::to_vec(&header)?;
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(&header);
        file.extend_from_slice(&data);
        Ok(std::fs::write(dir.join("weights.safetensors"), file)?)
    }

    /// Named weights load unread, take their bytes from the source they are linked to, and one
    /// the source does not have stays unread, failing the run that needs it.
    #[test]
    fn linked_weights_are_read_from_the_file() -> TractResult<()> {
        let dir = temp_dir::TempDir::new()?;
        let w = tensor1(&[f16::from_f32(1.5), f16::from_f32(-2.0)]);
        safetensors(dir.path(), &[("w", "F16", &w)])?;
        let graph = "version 1.0;\nextension tract_registry tract_core;\n\
            graph network() -> (w, v) {\n  \
            w = tract_core_named_weight(label = \"w\", shape = [2], datum_type = \"F16\");\n  \
            u = tract_core_named_weight(label = \"u\", shape = [3], datum_type = \"I64\");\n  \
            v = add(u, u);\n}\n";
        let proto = ProtoModel {
            doc: parse_document(graph)?,
            tensors: Default::default(),
            quantization: None,
            resources: Default::default(),
        };
        let mut model = crate::nnef().model_for_proto_model(&proto)?;
        let source = SafetensorsSource::open(dir.path().join("weights.safetensors"))?;
        LinkWeights(Arc::new(source)).transform(&mut model)?;
        let weight = |label: &str| {
            model
                .nodes()
                .iter()
                .filter_map(|n| n.op_as::<Const>())
                .find(|k| k.val().storage_as::<NamedStorage>().unwrap().name() == label)
                .unwrap()
                .val()
                .clone()
        };
        let read = weight("w");
        assert_eq!(
            read.try_as_plain_ram()?.as_slice::<f16>()?,
            w.try_as_plain_ram()?.as_slice::<f16>()?
        );
        assert!(!weight("u").storage_as::<NamedStorage>().unwrap().is_materialized());
        let err = model.into_runnable()?.run(tvec!()).unwrap_err();
        assert!(format!("{err:?}").contains("\"u\" has no source"), "{err:?}");
        Ok(())
    }
}
