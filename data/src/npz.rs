//! Reading and writing numpy `.npz` archives: zip files of `.npy` entries.

use std::io::{Read, Seek, Write};

use crate::internal::*;
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

use crate::npy::{npy_descr, read_npy, write_npy};

/// Reads tensors out of a `.npz` archive, as written by `numpy.savez` or
/// `numpy.savez_compressed`.
pub struct NpzReader<R: Read + Seek> {
    zip: ZipArchive<R>,
}

impl<R: Read + Seek> NpzReader<R> {
    pub fn new(reader: R) -> TractResult<NpzReader<R>> {
        Ok(NpzReader { zip: ZipArchive::new(reader)? })
    }

    /// Entry names, in archive order, with their `.npy` suffix stripped like numpy does.
    pub fn names(&self) -> TractResult<Vec<String>> {
        self.zip
            .file_names()
            .map(|n| {
                let n = n?;
                Ok(n.strip_suffix(".npy").unwrap_or(&n).to_string())
            })
            .collect()
    }

    /// Reads the entry `name`, falling back to `name.npy` like numpy does.
    ///
    /// The tensor allocation is bounded by the entry's uncompressed size as recorded in the
    /// archive, so a forged entry can not make it larger than a genuine one could.
    pub fn get(&mut self, name: &str) -> TractResult<Tensor> {
        let name = if self.zip.index_for_name(name).is_some() {
            name.to_string()
        } else {
            format!("{name}.npy")
        };
        let entry = self.zip.by_name(&name).with_context(|| format!("Reading {name} in npz"))?;
        let len = entry.size();
        read_npy(entry, len).with_context(|| format!("Reading {name} in npz"))
    }
}

/// Writes tensors as a `.npz` archive, readable with `numpy.load`.
///
/// The archive is completed by [`NpzWriter::finish`], or on drop with errors ignored.
pub struct NpzWriter<W: Write + Seek> {
    zip: ZipWriter<W>,
    options: SimpleFileOptions,
}

impl<W: Write + Seek> NpzWriter<W> {
    /// An archive of uncompressed entries, like `numpy.savez`.
    pub fn new(writer: W) -> NpzWriter<W> {
        Self::with_compression(writer, CompressionMethod::Stored)
    }

    /// An archive of deflated entries, like `numpy.savez_compressed`.
    pub fn new_compressed(writer: W) -> NpzWriter<W> {
        Self::with_compression(writer, CompressionMethod::Deflated)
    }

    fn with_compression(writer: W, method: CompressionMethod) -> NpzWriter<W> {
        let options = SimpleFileOptions::default().compression_method(method).large_file(true);
        NpzWriter { zip: ZipWriter::new(writer), options }
    }

    /// Adds `tensor` as entry `name.npy`. Fails on datum types npy can not represent.
    pub fn add(&mut self, name: &str, tensor: &Tensor) -> TractResult<()> {
        npy_descr(tensor.datum_type())
            .with_context(|| format!("Can not write {:?} tensor as npy", tensor.datum_type()))?;
        self.zip.start_file(format!("{name}.npy"), self.options)?;
        write_npy(&mut self.zip, tensor)
    }

    pub fn finish(self) -> TractResult<W> {
        Ok(self.zip.finish()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn round_trip() {
        for compressed in [false, true] {
            let buf = Cursor::new(vec![]);
            let mut npz =
                if compressed { NpzWriter::new_compressed(buf) } else { NpzWriter::new(buf) };
            npz.add("a", &tensor2(&[[1f32, 2.], [3., 4.]])).unwrap();
            npz.add("turn_0/b", &tensor1(&[f16::from_f32(0.5)])).unwrap();
            assert!(npz.add("c", &tensor0(TDim::from(1))).is_err());
            let bytes = npz.finish().unwrap().into_inner();

            let mut npz = NpzReader::new(Cursor::new(bytes)).unwrap();
            assert_eq!(npz.names().unwrap(), vec!["a", "turn_0/b"]);
            assert_eq!(npz.get("a").unwrap(), tensor2(&[[1f32, 2.], [3., 4.]]));
            assert_eq!(npz.get("turn_0/b.npy").unwrap(), tensor1(&[f16::from_f32(0.5)]));
            assert!(npz.get("c").is_err());
        }
    }
}
