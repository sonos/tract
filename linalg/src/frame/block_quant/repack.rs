use crate::mmm::MMMInputValue;

use super::*;
use crate::frame::PackedFormat;

#[derive(Clone, Debug, Hash)]
pub struct RepackingPackedBlockQuantValue {
    pub value: PackedBlockQuantValue,
    pub pack: PackedFormat,
}

impl Display for RepackingPackedBlockQuantValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?} repacked to {:?}", self.value, self.pack)
    }
}

impl MMMInputValue for RepackingPackedBlockQuantValue {
    fn scratch_panel_buffer_layout(&self) -> Option<Layout> {
        Some(self.pack.single_panel_layout(self.value.k, 4))
    }
    fn panel_bytes(&self, i: usize, buffer: Option<*mut u8>) -> TractResult<*const u8> {
        let buffer = buffer.context("Scratch panel expected")?;
        unsafe { self.value.format.repack_panel(&self.value, &self.pack, i, buffer)? };
        Ok(buffer)
    }
    fn mn(&self) -> usize {
        self.value.mn
    }
    fn r(&self) -> usize {
        self.pack.r
    }
    fn k(&self) -> usize {
        self.value.k
    }
}
