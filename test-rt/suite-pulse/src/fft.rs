use infra::{Test, TestSuite};
use proptest::prelude::*;
use tract_core::internal::*;
use tract_core::ndarray::ArrayD;
use tract_core::ops::fft::Fft;
use tract_core::runtime::Runtime;
use tract_pulse::internal::PulsedModel;
use tract_pulse::model::PulsedModelExt;

use crate::pulse_and_compare;

/// FFT applied on a non-streaming axis must be pulsifiable: the batch axes are
/// 1-to-1 passthrough in `Fft::axes_mapping`, so the generic per-pulse wrapper
/// handles streaming any of them. Without that mapping the pulse pass bails on
/// the first FFT it sees, which blocks the whole DPDFNet / DeepFilterNet
/// streaming family: STFT lowers to STFT + per-frame FFT, and the frame axis is
/// the streaming one while the FFT axis is the per-frame frequency axis.
///
/// The streaming axis is `rank - 3`, the FFT axis `rank - 2`, and the trailing
/// axis holds the (re, im) pair. Rank 3 is (stream, fft, 2), rank 4 prepends a
/// batch, rank 5 a batch and a channel.
#[derive(Clone, Debug)]
pub struct FftProblem {
    pub input_len: usize,
    pub pulse: usize,
    pub fft_size: usize,
    pub inverse: bool,
    pub rank: usize,
}

impl Arbitrary for FftProblem {
    type Parameters = bool;
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(inverse: bool) -> BoxedStrategy<Self> {
        (1usize..16, 1usize..4, proptest::sample::select(vec![2usize, 4, 8, 16]))
            .prop_map(move |(input_len, pulse, fft_size)| FftProblem {
                input_len,
                pulse,
                fft_size,
                inverse,
                rank: 4,
            })
            .boxed()
    }
}

impl Test for FftProblem {
    fn run_with_approx(
        &self,
        _id: &str,
        runtime: &dyn Runtime,
        approx: Approximation,
    ) -> infra::TestResult {
        ensure!(self.rank >= 3, "rank < 3 has no room for stream + fft + complex axes");
        let mut model = TypedModel::default();
        let s = model.symbols.sym("S");
        let mut sym_shape: Vec<TDim> = (0..self.rank - 3).map(|_| 1.to_dim()).collect();
        sym_shape.push(s.clone().into());
        sym_shape.push(self.fft_size.to_dim());
        sym_shape.push(2.to_dim());
        let a = model.add_source("a", f32::fact(&*sym_shape))?;
        model.wire_node("fft", Fft { axis: self.rank - 2, inverse: self.inverse }, &[a])?;
        model.auto_outputs()?;

        let mut shape = vec![1usize; self.rank - 3];
        shape.push(self.input_len);
        shape.push(self.fft_size);
        shape.push(2);
        let fft_size = self.fft_size;
        let rank = self.rank;
        let input: ArrayD<f32> = ArrayD::from_shape_fn(shape, |idx| {
            (idx[rank - 3] * fft_size * 2 + idx[rank - 2] * 2 + idx[rank - 1]) as f32 * 0.01
        });
        pulse_and_compare(runtime, approx, model, self.pulse, input, self.rank - 3)
    }
}

/// Stacked FFT then inverse FFT on the same axis: the streaming axis must be
/// tracked through both ops via the natural mapping.
#[derive(Clone, Debug)]
pub struct FftIfftRoundtrip;

impl Test for FftIfftRoundtrip {
    fn run_with_approx(
        &self,
        _id: &str,
        runtime: &dyn Runtime,
        approx: Approximation,
    ) -> infra::TestResult {
        let mut model = TypedModel::default();
        let s = model.symbols.sym("S");
        let a = model.add_source("a", f32::fact(dims!(1, s, 4, 2)))?;
        let fwd = model.wire_node("fft", Fft { axis: 2, inverse: false }, &[a])?;
        model.wire_node("ifft", Fft { axis: 2, inverse: true }, &fwd)?;
        model.auto_outputs()?;

        let input: ArrayD<f32> = ArrayD::from_shape_fn(vec![1, 8, 4, 2], |idx| {
            (idx[1] * 8 + idx[2] * 2 + idx[3]) as f32
        });
        pulse_and_compare(runtime, approx, model, 2, input, 1)
    }
}

/// Streaming on the FFT axis itself must be rejected: a per-pulse FFT on the
/// FFT axis is meaningless. `Fft::axes_mapping` declares that axis input-only,
/// so the generic pulse fallback can not track streaming through it and bails.
#[derive(Clone, Debug)]
pub struct StreamingOnFftAxisRefused;

impl Test for StreamingOnFftAxisRefused {
    fn run_with_approx(
        &self,
        _id: &str,
        _rt: &dyn Runtime,
        _a: Approximation,
    ) -> infra::TestResult {
        let mut model = TypedModel::default();
        let s = model.symbols.sym("S");
        let a = model.add_source("a", f32::fact(dims!(1, s, 2)))?;
        model.wire_node("fft", Fft { axis: 1, inverse: false }, &[a])?;
        model.auto_outputs()?;
        let err = PulsedModel::new(&model, s.clone(), &2.to_dim()).err();
        let msg = err.map(|e| format!("{e:#}")).unwrap_or_default();
        ensure!(
            msg.contains("could not track pulsing axis"),
            "expected the generic pulse fallback to refuse tracking the FFT axis, got: {msg}"
        );
        Ok(())
    }
}

/// Streaming the trailing (re, im) axis is structurally impossible:
/// `Fft::output_facts` rejects any input whose trailing axis is not 2, so a
/// symbolic trailing dim trips the typed-model build before the pulsifier runs.
#[derive(Clone, Debug)]
pub struct StreamingOnComplexAxisRefused;

impl Test for StreamingOnComplexAxisRefused {
    fn run_with_approx(
        &self,
        _id: &str,
        _rt: &dyn Runtime,
        _a: Approximation,
    ) -> infra::TestResult {
        let mut model = TypedModel::default();
        let s = model.symbols.sym("S");
        let a = model.add_source("a", f32::fact(dims!(1, 4, s)))?;
        let err = model.wire_node("fft", Fft { axis: 1, inverse: false }, &[a]).err();
        let msg = err.map(|e| format!("{e:#}")).unwrap_or_default();
        ensure!(
            msg.contains("inner (last) dimension to be 2"),
            "expected the typed-model rejection of a symbolic trailing axis, got: {msg}"
        );
        Ok(())
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();
    suite.add_arbitrary::<FftProblem>("proptest_fft", false);
    suite.add_arbitrary::<FftProblem>("proptest_ifft", true);
    let case = |input_len, pulse, fft_size, inverse, rank| FftProblem {
        input_len,
        pulse,
        fft_size,
        inverse,
        rank,
    };
    suite.add_test("smoke_8_pulse2_size4", case(8, 2, 4, false, 4));
    suite.add_test("smoke_4_pulse1_size8", case(4, 1, 8, false, 4));
    suite.add_test("smoke_6_pulse3_size4", case(6, 3, 4, false, 4));
    suite.add_test("ifft_smoke_8_pulse2_size4", case(8, 2, 4, true, 4));
    suite.add_test("rank3", case(8, 2, 4, false, 3));
    suite.add_test("rank5", case(8, 2, 4, false, 5));
    suite.add_test("fft_ifft_roundtrip", FftIfftRoundtrip);
    suite.add_test("streaming_on_fft_axis_refused", StreamingOnFftAxisRefused);
    suite.add_test("streaming_on_complex_axis_refused", StreamingOnComplexAxisRefused);
    Ok(suite)
}
