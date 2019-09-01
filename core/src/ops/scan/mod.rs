use crate::internal::*;

mod codegen;
mod inference;
mod typed;

pub use inference::Inference;
pub use typed::Typed;

#[derive(Debug, Clone, new)]
pub enum InputMapping<C: Clone> {
    Full { slot: usize },
    State { initializer: StateInitializer },
    Scan { slot: usize, axis: usize, chunk: C },
}

impl<C: Clone> InputMapping<C> {
    pub fn as_state(&self) -> Option<&StateInitializer> {
        match self {
            InputMapping::State { initializer } => Some(initializer),
            _ => None,
        }
    }

    pub fn as_scan(&self) -> Option<(usize, usize, C)> {
        match self {
            InputMapping::Scan { slot, axis, chunk } => Some((*slot, *axis, chunk.clone())),
            _ => None,
        }
    }

    pub fn invisible(&self) -> bool {
        if let InputMapping::State { initializer: StateInitializer::Value(_) } = self {
            true
        } else {
            false
        }
    }
}

#[derive(Debug, Clone, new)]
pub struct OutputMapping<C: Clone, F: Clone> {
    pub full_slot: Option<usize>,
    pub axis: usize,
    pub chunk: C,
    pub full_dim_hint: Option<F>,
    pub last_value_slot: Option<usize>,
    pub state: bool,
}

impl<C: Clone, F: Clone> OutputMapping<C, F> {
    pub fn invisible(&self) -> bool {
        self.full_slot.is_none() && self.last_value_slot.is_none()
    }
}

#[derive(Debug, Clone, new)]
pub enum StateInitializer {
    FromInput(usize),
    Value(Arc<Tensor>),
}
