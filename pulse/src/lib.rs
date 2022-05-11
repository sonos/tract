#[macro_use]
pub mod macros;

pub mod fact;
pub mod model;
pub mod ops;

pub mod internal {
    pub use std::fmt;
    pub use tract_nnef::internal::*;
    pub use tract_pulse_opl::tract_nnef;

    pub use downcast_rs::Downcast;

    pub use crate::fact::{stream_dim, stream_symbol, PulsedFact};
    pub use crate::model::{PulsedModel, PulsedModelExt};
    pub use crate::ops::{OpPulsifier, PulsedOp};
    pub use tract_pulse_opl::op_pulse;
}

use std::ops::ControlFlow;

use internal::*;

pub use ops::PulsedOp;

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

fn tract_nnef_registry() -> Registry {
    let mut reg = tract_pulse_opl::tract_nnef_registry();
    ops::delay::register(&mut reg);
    reg.extensions.push(Box::new(decl_stream_symbol));
    reg
}

fn decl_stream_symbol(
    proto_model: &mut ModelBuilder,
    args: &[String],
) -> TractResult<ControlFlow<(), ()>> {
    if args[0] == "tract_pulse_streaming_symbol" {
        proto_model.symbols.push(crate::stream_symbol());
        Ok(ControlFlow::Break(()))
    } else {
        Ok(ControlFlow::Continue(()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_must_stream() {
        let mut model = TypedModel::default();
        let _a = model.add_source("a", f32::fact(&[1, 2, 3])).unwrap();
        model.auto_outputs().unwrap();
        assert!(PulsedModel::new(&model, 4).is_err());

        let mut model = TypedModel::default();
        let _a = model
            .add_source("a", f32::fact([1.to_dim(), stream_dim(), 3.to_dim()].as_ref()))
            .unwrap();
        model.auto_outputs().unwrap();
        let pulse = PulsedModel::new(&model, 4).unwrap();
        assert_eq!(
            *pulse.outlet_fact(OutletId::new(0, 0)).unwrap().to_typed_fact().unwrap(),
            f32::fact(&[1usize, 4, 3])
        );
    }

    #[test]
    fn test_immediate() {
        let mut model = TypedModel::default();
        let _a = model
            .add_source("a", f32::fact([stream_dim(), 2.to_dim(), 3.to_dim()].as_ref()))
            .unwrap();
        model.auto_outputs().unwrap();

        let pulse = PulsedModel::new(&model, 4).unwrap();

        assert_eq!(*pulse.input_fact(0).unwrap().to_typed_fact().unwrap(), f32::fact([4, 2, 3]));
        assert_eq!(*pulse.output_fact(0).unwrap().to_typed_fact().unwrap(), f32::fact([4, 2, 3]));
    }
}
