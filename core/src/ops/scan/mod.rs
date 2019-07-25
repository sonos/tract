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

    pub fn invisible(&self) -> bool {
        if let InputMapping::State { initializer: StateInitializer::Value(_) } = self {
            true
        } else {
            false
        }
    }
}

#[derive(Debug, Clone, new)]
pub enum OutputMapping<C: Clone> {
    State { slot: Option<usize> },
    Scan { slot: usize, axis: usize, chunk: C },
}

impl<C: Clone> OutputMapping<C> {
    pub fn invisible(&self) -> bool {
        if let OutputMapping::State { slot: None } = self {
            true
        } else {
            false
        }
    }
}

#[derive(Debug, Clone, new)]
pub enum StateInitializer {
    FromInput(usize),
    Value(Arc<Tensor>),
}

