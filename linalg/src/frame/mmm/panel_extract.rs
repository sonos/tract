use std::fmt::{Debug, Display};

use super::pack::PackedFormat;
use super::{EagerPackedInput, MMMInputFormat, MMMInputValue};

type Kernel = unsafe fn(input: *const u8, output: *mut u8, k: usize);

#[derive(new, Hash, Clone)]
pub struct PanelExtractFormat {
    pub name: String,
    pub from: Box<dyn MMMInputFormat>,
    pub to: PackedFormat,
    pub kernel: Kernel,
}

#[derive(Clone, Hash)]
pub struct PanelExtractInput {
    pub format: PanelExtractFormat,
    pub data: EagerPackedInput,
}

impl MMMInputValue for PanelExtractInput {
    fn scratch_panel_buffer_layout(&self) -> Option<std::alloc::Layout> {
        Some(self.format.to.single_panel_layout(self.data.k(), self.format.to.dt.size_of()))
    }
    fn panel_bytes(&self, i: usize, buffer: Option<*mut u8>) -> tract_data::TractResult<*const u8> {
        let scratch = buffer.unwrap();
        unsafe {
            let source = self.data.packed.as_ptr().add(self.data.panel_bytes * i);
            (self.format.kernel)(source, scratch, self.data.k());
        }
        Ok(scratch)
    }
    fn mn(&self) -> usize {
        self.data.mn()
    }
    fn k(&self) -> usize {
        self.data.k()
    }
    fn format(&self) -> &dyn MMMInputFormat {
        &self.format.to
    }
}

impl Display for PanelExtractInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PanelExtract({})", self.data)
    }
}

impl Debug for PanelExtractInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PanelExtract({})", self.data)
    }
}
