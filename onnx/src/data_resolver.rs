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
        let length = length.unwrap_or(file_size - offset);
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
        match length {
            Some(length) => buf.extend_from_slice(&mmap[offset..offset + length]),
            None => buf.extend_from_slice(&mmap[offset..]),
        }
        Ok(())
    }
}
