use tract_nnef::internal::*;

#[macro_use]
mod macros;

mod concat;
mod delay;
mod pad;

pub use tract_nnef;
pub use tract_nnef::tract_core;

pub mod prelude {
    pub use crate::WithPulse;
    pub use tract_nnef::tract_core::internal::DimLike;
}

pub mod ops {
    pub use super::delay::Delay;
    pub use super::pad::PulsePad;
}

pub trait WithPulse {
    fn with_pulse(self) -> Self;
}

impl WithPulse for tract_nnef::framework::Nnef {
    fn with_pulse(mut self) -> Self {
        self = self.with_tract_core();
        self.registries.push(tract_nnef_registry());
        self
    }
}

pub fn tract_nnef_registry() -> Registry {
    let mut reg = Registry::new("pulse");
    delay::register(&mut reg);
    reg
}
