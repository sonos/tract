use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use tract_hir::internal::*;

use tract_hir::internal::TractResult;

#[cfg(not(target_family = "wasm"))]
pub fn default() -> Box<dyn ModelDataResolver> {
    Box::new(MmapDataResolver)
}

#[cfg(target_family = "wasm")]
pub fn default() -> Box<dyn ModelDataResolver> {
    Box::new(FopenDataResolver)
}

pub trait ModelDataResolver {
    fn read_bytes_from_path(
        &self,
        buf: &mut Vec<u8>,
        p: &Path,
        offset: usize,
        length: Option<usize>,
    ) -> TractResult<()>;

    /// Appends the bytes of an external tensor to `buf`: `length` bytes (to the end when
    /// `None`) from `offset` in `location`, as a TensorProto names it relative to the
    /// model's `model_dir`. The default resolves `location` to a file inside `model_dir`,
    /// refusing any path that escapes it, and reads it with
    /// [`read_bytes_from_path`](Self::read_bytes_from_path); a resolver with no file system
    /// behind it serves `location` itself.
    fn read_external(
        &self,
        buf: &mut Vec<u8>,
        model_dir: &str,
        location: &str,
        offset: usize,
        length: Option<usize>,
    ) -> TractResult<()> {
        let p = crate::tensor::resolve_external_data_path(Path::new(model_dir), location)?;
        self.read_bytes_from_path(buf, &p, offset, length)
    }
}

pub struct FopenDataResolver;

impl ModelDataResolver for FopenDataResolver {
    fn read_bytes_from_path(
        &self,
        buf: &mut Vec<u8>,
        p: &Path,
        offset: usize,
        length: Option<usize>,
    ) -> TractResult<()> {
        let file = File::open(p).with_context(|| format!("Opening {p:?}"))?;
        let file_size = file.metadata()?.len() as usize;
        ensure!(
            offset <= file_size,
            "external data offset {offset} is past end of file ({file_size} bytes)"
        );
        let length = length.unwrap_or(file_size - offset);
        ensure!(
            length <= file_size - offset,
            "external data length {length} from offset {offset} exceeds file size {file_size}"
        );
        buf.reserve(length);

        let mut reader = BufReader::new(file);
        reader.seek_relative(offset as i64)?;
        while reader.fill_buf()?.len() > 0 {
            let num_read = std::cmp::min(reader.buffer().len(), length - buf.len());
            buf.extend_from_slice(&reader.buffer()[..num_read]);
            if buf.len() == length {
                break;
            }
            reader.consume(reader.buffer().len());
        }
        Ok(())
    }
}

pub struct MmapDataResolver;

impl ModelDataResolver for MmapDataResolver {
    fn read_bytes_from_path(
        &self,
        buf: &mut Vec<u8>,
        p: &Path,
        offset: usize,
        length: Option<usize>,
    ) -> TractResult<()> {
        let file = File::open(p).with_context(|| format!("Opening {p:?}"))?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let end = match length {
            Some(length) => {
                offset.checked_add(length).context("external data offset + length overflows")?
            }
            None => mmap.len(),
        };
        ensure!(
            offset <= end && end <= mmap.len(),
            "external data range {offset}..{end} is out of bounds ({} bytes)",
            mmap.len()
        );
        buf.extend_from_slice(&mmap[offset..end]);
        Ok(())
    }
}
