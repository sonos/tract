use crate::internal::*;
use std::fmt;

mod lir;
mod mir;

pub use lir::{LirScan, State};
pub use mir::Scan;

#[derive(Clone, new, Hash, Eq, PartialEq, Copy, Debug)]
pub struct ScanInfo {
    pub slot: usize,
    pub axis: usize,
    pub chunk: isize,
}

#[derive(Clone, new, Hash)]
pub enum InputMapping {
    Full { slot: usize },
    State { init_slot: usize },
    Scan(ScanInfo),
}

impl InputMapping {
    pub fn as_state(&self) -> Option<usize> {
        match self {
            InputMapping::State { init_slot: initializer } => Some(*initializer),
            _ => None,
        }
    }

    pub fn as_scan(&self) -> Option<&ScanInfo> {
        match self {
            InputMapping::Scan(s) => Some(s),
            _ => None,
        }
    }

    pub fn outer_slot(&self) -> usize {
        match self {
            InputMapping::Full { slot } => *slot,
            InputMapping::Scan(info) => info.slot,
            InputMapping::State { init_slot: initializer } => *initializer
        }
    }
}

impl fmt::Debug for InputMapping {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self {
            InputMapping::Full { slot } => write!(fmt, "Full, inlet {slot}"),
            InputMapping::State { init_slot: initializer } => {
                write!(fmt, "State initialized by input {initializer}")
            }
            InputMapping::Scan(info) => {
                write!(
                    fmt,
                    "Scan inlet {}, axis: {}, chunk: {:?}.",
                    info.slot, info.axis, info.chunk
                )
            }
        }
    }
}

#[derive(Clone, new, Hash, Default)]
pub struct OutputMapping<F: Clone> {
    pub scan: Option<ScanInfo>,
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
        if let Some(info) = self.scan {
            write!(fmt, "Full value to outlet {} (axis: {}). ", info.slot, info.axis)?;
        }
        if let Some(full_dim_hint) = &self.full_dim_hint {
            write!(fmt, "Full len {full_dim_hint}. ")?;
        }
        Ok(())
    }
}
