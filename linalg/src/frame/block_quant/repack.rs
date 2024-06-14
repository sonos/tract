use crate::mmm::MMMInputValue;

use super::*;
use crate::frame::PackedFormat;

#[derive(Clone, Hash)]
pub struct RepackingPackedBlockQuantValue {
    pub value: EagerPackedInput,
    pub pack: PackedFormat,
}

impl Display for RepackingPackedBlockQuantValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?} repacked to {:?}", self.value, self.pack)
    }
}

impl Debug for RepackingPackedBlockQuantValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        <Self as Display>::fmt(self, f)
    }
}

impl MMMInputValue for RepackingPackedBlockQuantValue {
    fn scratch_panel_buffer_layout(&self) -> Option<Layout> {
        Some(self.pack.single_panel_layout(self.value.k, 4))
    }
    fn panel_bytes(&self, _i: usize, _buffer: Option<*mut u8>) -> TractResult<*const u8> {
        todo!()
        /*
        let buffer = buffer.context("Scratch panel expected")?;
        unsafe { self.value.format.repack_panel(&self.value, &self.pack, i, buffer)? };
        Ok(buffer)
        */
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
