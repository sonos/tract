use crate::mmm::MMMInputValue;

use super::*;
use crate::frame::Packer;

#[derive(Clone, Debug, Hash)]
pub struct RepackingPackedBlockQuantValue {
    pub value: PackedBlockQuantValue,
    pub pack: Packer,
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
    fn panel_bytes(&self, i: usize, buffer: Option<*mut u8>) -> *const u8 {
        let buffer = buffer.unwrap();
        let scratch = unsafe {
            std::slice::from_raw_parts_mut(
                buffer as *mut f32,
                self.pack.single_panel_len(self.value.k),
            )
        };
        self.value.format.panel_f32(
            &self.value.packed_block_quant_data,
            self.value.k,
            self.pack.r,
            i,
            scratch,
        );
        buffer
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
