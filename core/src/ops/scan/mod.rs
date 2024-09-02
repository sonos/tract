use crate::internal::*;
use std::fmt;

mod decluttered;
mod optimized;

pub use optimized::{OptScan, State};
pub use decluttered::Scan;

#[derive(Clone, new, Hash, Eq, PartialEq, Copy, Debug)]
pub struct ScanInfo {
    pub axis: usize,
    pub chunk: isize,
}

#[derive(Clone, new, Hash, Debug)]
pub enum InputMapping {
    Full,
    State,
    Scan(ScanInfo),
}

impl InputMapping {
    pub fn is_state(&self) -> bool {
        matches!(self, InputMapping::State)
    }

    pub fn is_scan(&self) -> bool {
        self.as_scan().is_some()
    }

    pub fn as_scan(&self) -> Option<&ScanInfo> {
        match self {
            InputMapping::Scan(s) => Some(s),
            _ => None,
        }
    }
}

#[derive(Clone, new, Hash, Default)]
pub struct OutputMapping<F: Clone> {
    pub scan: Option<(usize, ScanInfo)>,
    pub full_dim_hint: Option<F>,
    pub last_value_slot: Option<usize>,
    pub state: bool,
}

impl<F: Clone> OutputMapping<F> {
    pub fn invisible(&self) -> bool {
        self.scan.is_none() && self.last_value_slot.is_none()
    }
}

impl<F: Clone + DimLike> OutputMapping<F> {
    pub fn concretize_dims(&self, values: &SymbolValues) -> TractResult<OutputMapping<F>> {
        Ok(Self {
            full_dim_hint: self.full_dim_hint.as_ref().map(|h| h.eval(values)),
            ..self.clone()
        })
    }
}

impl<F: Clone + fmt::Display> fmt::Debug for OutputMapping<F> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        if self.state {
            write!(fmt, "State. ")?;
        }
        if let Some(last_value_slot) = self.last_value_slot {
            write!(fmt, "Last value to outlet {last_value_slot}. ")?;
        }
        if let Some((slot, info)) = self.scan {
            write!(fmt, "Full value to outlet {} (axis: {}). ", slot, info.axis)?;
        }
        if let Some(full_dim_hint) = &self.full_dim_hint {
            write!(fmt, "Full len {full_dim_hint}. ")?;
        }
        Ok(())
    }
}

pub fn iteration_count(input_mapping: &[InputMapping], inputs: &[&TypedFact]) -> Option<TDim> {
    let (slot, info) = input_mapping
        .iter()
        .enumerate()
        .find_map(|(slot, im)| im.as_scan().map(|scan| (slot, scan)))?;
    let outside_dim = inputs[slot].shape[info.axis].clone();
    Some(outside_dim.div_ceil(info.chunk.unsigned_abs() as u64))
}
