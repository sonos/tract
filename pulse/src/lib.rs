#![allow(clippy::len_zero)]
#[macro_use]
pub mod macros;

pub mod blockify;
pub mod fact;
pub mod model;
pub mod ops;

pub mod internal {
    pub use std::fmt;
    pub use tract_nnef::internal::*;
    pub use tract_pulse_opl::tract_nnef;

    pub use downcast_rs::Downcast;

    pub use crate::fact::PulsedFact;
    pub use crate::model::{PulsedModel, PulsedModelExt};
    pub use crate::ops::{OpPulsifier, PulsedOp};
}

use std::ops::ControlFlow;

use internal::*;
use tract_core::optim::TypedPass;
use tract_core::transform::ModelTransform;
use tract_pulse_opl::tract_nnef::tract_core;

pub use ops::PulsedOp;

/// Model property listing input names whose value is fed once per session
/// (rather than chunked over pulses), even if their shape contains the
/// streaming symbol.  Used for things like Transformer-XL relative-position
/// tables: they're indexed dynamically per chunk, and chunk c reads from
/// pos rows that wouldn't have arrived yet under a forward stream.  The
/// resolution is to bind the full pos tensor at session start (when the
/// streaming symbol is concretely known).
///
/// Stored as a list-of-strings tensor of input node names.  The Source
/// pulsifier consults this list and forces `stream: None` for matching
/// inputs.  Blockify populates it when its rel-pos einsum→DiagGather
/// rewrite identifies a streaming-rel-pos input.
pub const PULSE_SESSION_BUFFERED_INPUTS: &str = "pulse.session_buffered_inputs";

#[derive(Debug, Default, serde::Deserialize)]
pub struct PulseConfig {
    pub symbol: Option<String>,
    pub pulse: String,
}

#[derive(Debug)]
struct PulseTransform(PulseConfig);

impl ModelTransform for PulseTransform {
    fn name(&self) -> std::borrow::Cow<'static, str> {
        "pulse".into()
    }
    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        let symbol = self.0.symbol.as_deref().unwrap_or("S");
        let sym = model.symbols.sym(symbol);
        let pulse_dim = parse_tdim(&model.symbols, &self.0.pulse)?;
        // Pre-pulsification: fold skew-trick chains into DiagGather.
        if ops::diag_gather::fold_diag_gather(model)? {
            // Re-run ROI propagation: the fold replaces Slice nodes that
            // carried ROI annotations, so the new DiagGather node needs
            // its own ROI annotation re-derived from downstream consumers.
            tract_core::optim::propagate_roi::PropagateRoi.run_direct(model)?;
            // Re-declutter: the fold leaves the original Pad/Reshape/Slice
            // nodes shunted-out but still in the graph, where blockify's
            // section recogniser would see them as multi-T-axis dead nodes
            // and bail.  Mirrors the post-`-t` declutter the CLI runs
            // between explicit transform stages.
            model.declutter()?;
        }
        // Pulsification (which calls Blockify internally) consumes the
        // model and produces a typed pulsed graph.
        let pulsed = model::PulsedModel::new(model, sym, &pulse_dim)?;
        *model = pulsed.into_typed()?;
        Ok(())
    }
}

register_model_transform!("pulse", PulseConfig, |config| Ok(Box::new(PulseTransform(config))));

register_model_transform!("blockify", blockify::BlockifyConfig, |config| Ok(Box::new(
    blockify::BlockifyTransform(config)
)));

#[derive(Debug, Default, serde::Deserialize)]
pub struct DiagGatherFoldConfig {}

#[derive(Debug)]
struct DiagGatherFoldTransform;

impl ModelTransform for DiagGatherFoldTransform {
    fn name(&self) -> std::borrow::Cow<'static, str> {
        "diag_gather_fold".into()
    }
    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        if ops::diag_gather::fold_diag_gather(model)? {
            tract_core::optim::propagate_roi::PropagateRoi.run_direct(model)?;
        }
        Ok(())
    }
}

register_model_transform!("diag_gather_fold", DiagGatherFoldConfig, |_config| Ok(Box::new(
    DiagGatherFoldTransform
)));

pub trait WithPulse {
    fn enable_pulse(&mut self);
    fn with_pulse(self) -> Self;
}

impl WithPulse for tract_nnef::framework::Nnef {
    fn enable_pulse(&mut self) {
        self.registries.push(tract_nnef_registry());
    }
    fn with_pulse(mut self) -> Self {
        self.enable_pulse();
        self
    }
}

pub fn tract_nnef_registry() -> Registry {
    let mut reg = tract_pulse_opl::tract_nnef_registry();
    ops::delay::register(&mut reg);
    reg.extensions.push(Box::new(decl_stream_symbol));
    reg
}

fn decl_stream_symbol(
    _proto_model: &mut ModelBuilder,
    name: &Identifier,
    _rest: &str,
) -> TractResult<ControlFlow<(), ()>> {
    if name.0 == "tract_pulse_streaming_symbol" {
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
        let s = model.symbols.sym("S");
        let _a = model.add_source("a", f32::fact([1, 2, 3])).unwrap();
        model.auto_outputs().unwrap();
        assert!(PulsedModel::new(&model, s.clone(), &4.to_dim()).is_err());

        let mut model = TypedModel::default();
        let _a = model.add_source("a", f32::fact(dims![1, s, 3].as_ref())).unwrap();
        model.auto_outputs().unwrap();
        let pulse = PulsedModel::new(&model, s, &4.to_dim()).unwrap();
        assert_eq!(
            *pulse.outlet_fact(OutletId::new(0, 0)).unwrap().to_typed_fact().unwrap(),
            f32::fact([1usize, 4, 3])
        );
    }

    #[test]
    fn test_immediate() {
        let mut model = TypedModel::default();
        let s = model.symbols.sym("S");
        let _a = model.add_source("a", f32::fact(dims![s, 2, 3].as_ref())).unwrap();
        model.auto_outputs().unwrap();

        let pulse = PulsedModel::new(&model, s, &4.to_dim()).unwrap();

        assert_eq!(*pulse.input_fact(0).unwrap().to_typed_fact().unwrap(), f32::fact([4, 2, 3]));
        assert_eq!(*pulse.output_fact(0).unwrap().to_typed_fact().unwrap(), f32::fact([4, 2, 3]));
    }

    #[test]
    fn test_reshape_split_streaming_axis() {
        use tract_core::ops::change_axes::AxisOp;
        let mut model = TypedModel::default();
        let s = model.symbols.sym("S");
        let a = model.add_source("a", f32::fact(dims![s.to_dim() * 2, 4].as_ref())).unwrap();
        let split = model
            .wire_node(
                "split",
                AxisOp::Reshape(0, tvec!(s.to_dim() * 2), tvec!(s.to_dim(), 2.to_dim())),
                &[a],
            )
            .unwrap();
        model.select_output_outlets(&split).unwrap();
        let pulse = PulsedModel::new(&model, s.clone(), &1.to_dim()).unwrap();
        assert_eq!(*pulse.input_fact(0).unwrap().to_typed_fact().unwrap(), f32::fact([2, 4]));
        assert_eq!(*pulse.output_fact(0).unwrap().to_typed_fact().unwrap(), f32::fact([1, 2, 4]));
        let out_stream = pulse.output_fact(0).unwrap().stream.as_ref().unwrap();
        assert_eq!(out_stream.axis, 0);
        assert_eq!(out_stream.dim, s.to_dim());
    }

    #[test]
    fn test_reshape_merge_streaming_axis() {
        use tract_core::ops::change_axes::AxisOp;
        let mut model = TypedModel::default();
        let s = model.symbols.sym("S");
        let a = model.add_source("a", f32::fact(dims![s, 2, 4].as_ref())).unwrap();
        let merged = model
            .wire_node(
                "merge",
                AxisOp::Reshape(0, tvec!(s.to_dim(), 2.to_dim()), tvec!(s.to_dim() * 2)),
                &[a],
            )
            .unwrap();
        model.select_output_outlets(&merged).unwrap();
        let pulse = PulsedModel::new(&model, s.clone(), &1.to_dim()).unwrap();
        assert_eq!(*pulse.input_fact(0).unwrap().to_typed_fact().unwrap(), f32::fact([1, 2, 4]));
        assert_eq!(*pulse.output_fact(0).unwrap().to_typed_fact().unwrap(), f32::fact([2, 4]));
        let out_stream = pulse.output_fact(0).unwrap().stream.as_ref().unwrap();
        assert_eq!(out_stream.axis, 0);
        assert_eq!(out_stream.dim, s.to_dim() * 2);
    }

    #[test]
    fn test_reshape_split_then_run() {
        use tract_core::ops::change_axes::AxisOp;
        let mut model = TypedModel::default();
        let s = model.symbols.sym("S");
        let a = model.add_source("a", f32::fact(dims![s.to_dim() * 2].as_ref())).unwrap();
        let split = model
            .wire_node(
                "split",
                AxisOp::Reshape(0, tvec!(s.to_dim() * 2), tvec!(s.to_dim(), 2.to_dim())),
                &[a],
            )
            .unwrap();
        model.select_output_outlets(&split).unwrap();

        let pulse = PulsedModel::new(&model, s, &1.to_dim()).unwrap();
        let plan = SimplePlan::new(pulse.into_typed().unwrap()).unwrap();
        let mut state = SimpleState::new(&plan).unwrap();
        let chunk1 = tensor1(&[1f32, 2.0]);
        let out1 = state.run(tvec!(chunk1.into_tvalue())).unwrap();
        assert_eq!(*out1[0], tensor2(&[[1f32, 2.0]]).into());
        let chunk2 = tensor1(&[3f32, 4.0]);
        let out2 = state.run(tvec!(chunk2.into_tvalue())).unwrap();
        assert_eq!(*out2[0], tensor2(&[[3f32, 4.0]]).into());
    }
}
