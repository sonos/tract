use proptest::test_runner::TestCaseResult;
use tract_core::ops::fft::Fft;

use super::*;

/// FFT applied on a non-streaming axis must be pulsifiable: every axis
/// is a 1-to-1 passthrough, so the generic per-pulse wrapper handles it
/// once `Fft::axes_mapping` reports `natural`. Without that mapping the
/// pulse pass bails on the first FFT it sees, which blocks the entire
/// DPDFNet / DeepFilterNet streaming family (STFT lowers to STFT + per-
/// frame FFT, and the frame axis is the streaming one but the FFT axis
/// is the per-frame frequency axis: distinct, so streaming is sound).
///
/// Setup: input is rank-4 (batch, T_stream, fft_size, 2). T_stream is
/// the streaming axis; FFT runs on `fft_size`; trailing 2 holds
/// (re, im). Pulse processes one chunk along T_stream at a time.
fn fft_on_non_streaming_axis(input_len: usize, pulse: usize, fft_size: usize) -> TestCaseResult {
    let mut model = TypedModel::default();
    let s = model.symbols.sym("S");
    let a = model
        .add_source("a", f32::fact(dims!(1, s, fft_size, 2)))
        .unwrap();
    model
        .wire_node("fft", Fft { axis: 2, inverse: false }, &[a])
        .unwrap();
    model.auto_outputs().unwrap();

    let input: ArrayD<f32> =
        ArrayD::from_shape_fn(vec![1, input_len, fft_size, 2], |idx| {
            (idx[1] * fft_size * 2 + idx[2] * 2 + idx[3]) as f32 * 0.01
        });
    proptest_regular_against_pulse(model, pulse, input, 1)
}

#[test]
fn fft_pulse_smoke_8_pulse2_size4() {
    fft_on_non_streaming_axis(8, 2, 4).unwrap();
}

#[test]
fn fft_pulse_smoke_4_pulse1_size8() {
    fft_on_non_streaming_axis(4, 1, 8).unwrap();
}

#[test]
fn fft_pulse_smoke_6_pulse3_size4() {
    fft_on_non_streaming_axis(6, 3, 4).unwrap();
}

proptest! {
    #[test]
    fn proptest_fft_pulse(
        input_len in 1usize..16,
        pulse in 1usize..4,
        fft_size in proptest::sample::select(vec![2usize, 4, 8, 16]),
    ) {
        fft_on_non_streaming_axis(input_len, pulse, fft_size)?
    }
}
