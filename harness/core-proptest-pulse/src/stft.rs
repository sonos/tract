use proptest::test_runner::TestCaseResult;
use tract_core::ops::fft::Stft;

use super::*;

/// STFT applied with the streaming axis distinct from the STFT axis
/// must be pulsifiable: every non-STFT axis is a 1-to-1 passthrough
/// once `Stft::axes_mapping` declares the relationship (input axis
/// `op.axis` maps to output `op.axis` as `n_frames`; output `op.axis +
/// 1` is the inserted frame axis; the rest shift naturally). Without
/// the mapping the pulse pass bails with "could not track pulsing
/// axis" the moment a batched STFT pipeline streams its batch axis.
///
/// Setup: input is rank-3 `(B_stream, T, 2)`. B_stream is the
/// streaming axis (axis 0); STFT runs on the T axis (axis 1); the
/// trailing 2 holds (re, im). One pulse = one batch element; tract
/// runs the full-length STFT inside each pulse.
fn stft_on_non_stft_axis(
    batch_len: usize,
    pulse: usize,
    time_len: usize,
    frame: usize,
    stride: usize,
) -> TestCaseResult {
    let mut model = TypedModel::default();
    let s = model.symbols.sym("S");
    let a = model.add_source("a", f32::fact(dims!(s, time_len, 2))).unwrap();
    model.wire_node("stft", Stft { axis: 1, frame, stride, window: None }, &[a]).unwrap();
    model.auto_outputs().unwrap();

    let input: ArrayD<f32> = ArrayD::from_shape_fn(vec![batch_len, time_len, 2], |idx| {
        (idx[0] * time_len * 2 + idx[1] * 2 + idx[2]) as f32 * 0.01
    });
    proptest_regular_against_pulse(model, pulse, input, 0)
}

#[test]
fn stft_pulse_batch_axis_smoke_4_pulse2_t8_frame4_stride2() {
    stft_on_non_stft_axis(4, 2, 8, 4, 2).unwrap();
}

#[test]
fn stft_pulse_batch_axis_smoke_3_pulse1_t6_frame3_stride1() {
    stft_on_non_stft_axis(3, 1, 6, 3, 1).unwrap();
}

#[test]
fn stft_pulse_batch_axis_smoke_2_pulse2_t12_frame4_stride4() {
    stft_on_non_stft_axis(2, 2, 12, 4, 4).unwrap();
}

proptest! {
    #[test]
    fn proptest_stft_pulse_batch_axis(
        batch_len in 1usize..6,
        pulse in 1usize..3,
        time_len in 4usize..16,
        frame in proptest::sample::select(vec![2usize, 4]),
        stride in proptest::sample::select(vec![1usize, 2]),
    ) {
        // Skip frame > time_len -- the STFT would produce 0 frames.
        prop_assume!(time_len >= frame);
        stft_on_non_stft_axis(batch_len, pulse, time_len, frame, stride)?
    }
}
