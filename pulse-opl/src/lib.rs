use tract_nnef::internal::*;

pub mod concat;
mod deconv_delay;
mod delay;
mod mask;
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
    pub use super::mask::PulseMask;
    pub use super::pad::PulsePad;
    pub use super::slice::PulsedAxisSlice;
}

pub trait WithPulse {
    fn enable_pulse(&mut self);
    fn with_pulse(self) -> Self;
}

impl WithPulse for tract_nnef::framework::Nnef {
    fn enable_pulse(&mut self) {
        self.enable_tract_core();
        self.registries.push(tract_nnef_registry());
    }
    fn with_pulse(mut self) -> Self {
        self.enable_pulse();
        self
    }
}

pub fn tract_nnef_registry() -> Registry {
    let mut reg = Registry::new("tract_pulse")
        .with_doc("Extension `tract_resource` extends NNEF with operators")
        .with_doc("for pulsified networks.")
        .with_doc("")
        .with_doc("Add `extension tract_pulse` to `graph.nnef`");
        
    reg.aliases.push("pulse".into());
    delay::register(&mut reg);
    mask::register(&mut reg);
    pad::register(&mut reg);
    reg
}
