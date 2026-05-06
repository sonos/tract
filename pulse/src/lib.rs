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
        ops::diag_gather::detect_diag_gather(model)?;
        tract_core::optim::propagate_roi::PropagateRoi.run_direct(model)?;
        model.declutter()?;
        let pulsed = model::PulsedModel::new(model, sym, &pulse_dim)?;
        *model = pulsed.into_typed()?;
        Ok(())
    }
}

register_model_transform!("pulse", PulseConfig, |config| Ok(Box::new(PulseTransform(config))));

register_model_transform!("blockify", blockify::BlockifyConfig, |config| Ok(Box::new(
    blockify::BlockifyTransform(config)
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

    /// Two parallel pulse paths meeting at an elementwise op produce
    /// different per-pulse stream-axis sizes when one path goes through a
    /// ConvTranspose (kernel > stride) and the other doesn't.  Pre-fix
    /// pulsification bailed at the meet point because the typed
    /// `output_facts`' `multi_broadcast` returned `Broadcast(K_a, K_b)`
    /// on the stream axis -- not equal, not 1, doesn't simplify.  After fix
    /// the merge uses LCM for the stream axis specifically.
    ///
    /// Minimal repro of the Pocket-TTS upsample-then-attention pattern:
    /// a ConvTranspose1d(stride=4, kernel=8) emits steady-state stride=4
    /// frames per pulse with 4-frame overlap-add; an arange of the same
    /// post-convtr length produces (after our Range slope-based fix) also
    /// 4 frames per pulse; an elementwise Add of the two requires the
    /// meet-point merge to be LCM(4, 4) = 4 (trivial here, but the path
    /// went through Broadcast(4, 8) before slope+LCM fixes were in place).
    #[test]
    fn test_pulse_meet_with_arange_branch_types_through() {
        use tract_core::ops::array::Range;
        use tract_core::ops::cnn::{Deconv, KernelFormat, PaddingSpec, PoolSpec};
        use tract_core::ops::nn::DataFormat;

        let mut model = TypedModel::default();
        let t = model.symbols.sym("T");
        let src = model.add_source("x", f32::fact(dims![1, 2, t.to_dim()].as_ref())).unwrap();

        // ConvTranspose1d(C=2, kernel=8, stride=4) → stream-axis dim
        // becomes 4*T + 4 (post overlap-add tail).
        let kernel = model
            .add_const("kernel", tract_core::ndarray::Array3::<f32>::zeros((2, 2, 8)))
            .unwrap();
        let bias = model.add_const("bias", tract_core::ndarray::arr1(&[0.0f32, 0.0])).unwrap();
        let conv_out = model
            .wire_node(
                "convtr",
                Deconv {
                    pool_spec: PoolSpec {
                        data_format: DataFormat::NCHW,
                        kernel_shape: tvec!(8),
                        padding: PaddingSpec::Valid,
                        dilations: Some(tvec!(1)),
                        strides: Some(tvec!(4)),
                        input_channels: 2,
                        output_channels: 2,
                    },
                    kernel_format: KernelFormat::OIHW,
                    adjustments: tvec!(0),
                    group: 1,
                },
                &[src, kernel, bias],
            )
            .unwrap()[0];

        // arange(0, 4*T + 4) of the same stream-axis length — this is the
        // branch that surfaced the Broadcast bug pre-fix.
        let start = model.add_const("range_start", tensor0(TDim::Val(0))).unwrap();
        let end = model
            .add_const(
                "range_end",
                tract_core::ndarray::arr0(t.to_dim() * 4 + 4).into_dyn().into_tensor(),
            )
            .unwrap();
        let step = model.add_const("range_step", tensor0(TDim::Val(1))).unwrap();
        let range_out = model
            .wire_node("range", Range::new(t.to_dim() * 4 + 4), &[start, end, step])
            .unwrap()[0];

        // Cast range to f32 and broadcast-shape with conv_out so they Add.
        let range_f32 = model
            .wire_node("range_cast", tract_core::ops::cast::cast(f32::datum_type()), &[range_out])
            .unwrap()[0];
        let range_bc = model
            .wire_node(
                "range_unsqueeze",
                tract_core::ops::change_axes::AxisOp::Add(0),
                &[range_f32],
            )
            .unwrap()[0];
        let range_bc = model
            .wire_node(
                "range_unsqueeze2",
                tract_core::ops::change_axes::AxisOp::Add(0),
                &[range_bc],
            )
            .unwrap()[0];

        let added =
            model.wire_node("add", tract_core::ops::math::add(), &[conv_out, range_bc]).unwrap();
        model.select_output_outlets(&added).unwrap();

        // The point of the test: this used to panic with
        // `Pulsification requires pulse Broadcast(4, 8) ...` at the
        // downstream meet point.  Now it should pulsify without error.
        let _pulse = PulsedModel::new(&model, t, &2.to_dim()).expect("pulsification");
    }
}
