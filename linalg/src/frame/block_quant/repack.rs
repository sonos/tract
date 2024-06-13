use crate::mmm::MMMInputValue;

use super::*;
use crate::frame::Packer;


#[derive(Clone, Debug, Hash)]
pub struct RepackingPackedBlockQuantValue {
    pub format: Box<dyn BlockQuant>,
    pub packed_block_quant_data: Blob,
    pub pack: Packer,
    pub mn: usize,
    pub k: usize,
}

impl Display for RepackingPackedBlockQuantValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Packed{} (m={} k={} r={})", self.format, self.mn, self.k, self.pack.r)
    }
}

impl MMMInputValue for RepackingPackedBlockQuantValue {
    fn scratch_panel_buffer_layout(&self) -> Option<Layout> {
        Some(self.pack.single_panel_layout(self.k, 4))
    }
    fn panel_bytes(&self, i: usize, buffer: Option<*mut u8>) -> *const u8 {
        let buffer = buffer.unwrap();
        let scratch = unsafe {
            std::slice::from_raw_parts_mut(buffer as *mut f32, self.pack.single_panel_len(self.k))
        };
        self.format.panel_f32(&self.packed_block_quant_data, self.k, self.pack.r, i, scratch);
        buffer
    }
    fn mn(&self) -> usize {
        self.mn
    }
    fn r(&self) -> usize {
        self.pack.r
    }
    fn k(&self) -> usize {
        self.k
    }
}
