use std::fmt::{Debug, Display};

use tract_data::prelude::DatumType;

use crate::frame::PackedFormat;
use crate::mmm::{EagerPackedInput, MMMInputFormat, MMMInputValue};

use super::PackedBlockQuantFormat;

#[derive(Clone, Hash)]
pub struct PanelExtractFormat {
    pub pbqf: PackedBlockQuantFormat,
}

impl MMMInputFormat for PanelExtractFormat {
    fn prepare_tensor(
        &self,
        _t: &tract_data::prelude::Tensor,
        _k_axis: usize,
        _mn_axis: usize,
    ) -> tract_data::TractResult<Box<dyn crate::mmm::MMMInputValue>> {
        todo!()
    }

    fn k_alignment(&self) -> usize {
        self.pbqf.k_alignment()
    }

    fn r(&self) -> usize {
        self.pbqf.r
    }
}

impl Display for PanelExtractFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PanelExtract({})", self.pbqf)
    }
}

impl Debug for PanelExtractFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PanelExtract({})", self.pbqf)
    }
}

#[derive(Clone, Hash)]
pub struct PanelExtractInput {
    pbqf: PackedBlockQuantFormat,
    data: EagerPackedInput,
    to: PackedFormat,
    dt: DatumType,
}

impl MMMInputValue for PanelExtractInput {
    fn scratch_panel_buffer_layout(&self) -> Option<std::alloc::Layout> {
        Some(self.to.single_panel_layout(self.data.k(), self.dt.size_of()))
    }
    fn panel_bytes(&self, i: usize, buffer: Option<*mut u8>) -> tract_data::TractResult<*const u8> {
        let scratch = buffer.unwrap();
        unsafe {
            self.pbqf.bq.extract_panel(&self.data, &self.to, i, scratch)?;
        }
        Ok(scratch)
    }
    fn mn(&self) -> usize {
        self.data.mn()
    }
    fn r(&self) -> usize {
        self.data.r()
    }
    fn k(&self) -> usize {
        self.data.k()
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
