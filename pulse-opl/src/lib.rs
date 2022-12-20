use tract_nnef::internal::*;

mod concat;
mod deconv_delay;
mod delay;
mod pad;
mod slice;

pub use tract_nnef;
pub use tract_nnef::tract_core;

pub mod prelude {
    pub use crate::WithPulse;
    pub use tract_nnef::tract_core::internal::DimLike;
}

pub mod ops {
    pub use super::deconv_delay::DeconvDelay;
    pub use super::delay::{ Delay, DelayState };
    pub use super::pad::PulsePad;
    pub use super::slice::PulsedAxisSlice;
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
    let mut reg = Registry::new("tract_pulse");
    reg.aliases.push("pulse".to_string());
    delay::register(&mut reg);
    pad::register(&mut reg);
    reg
}
