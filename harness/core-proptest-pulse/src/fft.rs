use proptest::test_runner::TestCaseResult;
use tract_core::ops::fft::Fft;

use super::*;

/// FFT applied on a non-streaming axis must be pulsifiable: the batch
/// axes are 1-to-1 passthrough in `Fft::axes_mapping`, so the generic
/// per-pulse wrapper handles streaming any of them. Without that mapping
/// the pulse pass bails on the first FFT it sees, which blocks the entire
/// DPDFNet / DeepFilterNet streaming family (STFT lowers to STFT + per-
/// frame FFT, and the frame axis is the streaming one but the FFT axis
/// is the per-frame frequency axis: distinct, so streaming is sound).
///
/// Setup: input is rank-4 (batch, T_stream, fft_size, 2). T_stream is
/// the streaming axis; FFT runs on `fft_size`; trailing 2 holds
/// (re, im). Pulse processes one chunk along T_stream at a time.
fn fft_on_non_streaming_axis(input_len: usize, pulse: usize, fft_size: usize) -> TestCaseResult {
    fft_inner(input_len, pulse, fft_size, /* inverse = */ false, /* rank = */ 4)
}

fn ifft_on_non_streaming_axis(input_len: usize, pulse: usize, fft_size: usize) -> TestCaseResult {
    fft_inner(input_len, pulse, fft_size, /* inverse = */ true, /* rank = */ 4)
}

/// Generic harness: build a model where the streaming axis is `1`, the
/// FFT axis is `rank - 2`, and the trailing axis (rank-1) is the
/// (re, im) pair. Rank 3 = (stream, fft, 2); rank 4 = (batch=1, stream,
/// fft, 2); rank 5 = (B=1, C=1, stream, fft, 2). All three are
/// rank-preserving so axes_mapping::natural lines them up identically.
fn fft_inner(
    input_len: usize,
    pulse: usize,
    fft_size: usize,
    inverse: bool,
    rank: usize,
) -> TestCaseResult {
    assert!(rank >= 3, "rank < 3 has no room for stream + fft + complex axes");
    let mut model = TypedModel::default();
    let s = model.symbols.sym("S");
    // Build the symbolic input shape: leading 1s, then (S, fft_size, 2).
    let mut sym_shape: Vec<TDim> = (0..rank - 3).map(|_| 1.to_dim()).collect();
    sym_shape.push(s.clone().into());
    sym_shape.push(fft_size.to_dim());
    sym_shape.push(2.to_dim());
    let a = model.add_source("a", f32::fact(&*sym_shape)).unwrap();
    // FFT axis: the second-to-last (just before the (re, im) pair).
    let fft_axis = rank - 2;
    model.wire_node("fft", Fft { axis: fft_axis, inverse }, &[a]).unwrap();
    model.auto_outputs().unwrap();

    // Concrete input: leading 1s, then (input_len, fft_size, 2).
    let mut shape = vec![1usize; rank - 3];
    shape.push(input_len);
    shape.push(fft_size);
    shape.push(2);
    let input: ArrayD<f32> = ArrayD::from_shape_fn(shape, |idx| {
        // Some non-trivial pattern so the FFT outputs aren't all zeros.
        let stream_ix = idx[rank - 3];
        let freq_ix = idx[rank - 2];
        let cmplx_ix = idx[rank - 1];
        (stream_ix * fft_size * 2 + freq_ix * 2 + cmplx_ix) as f32 * 0.01
    });
    // Streaming axis = rank - 3 (the one we labelled S).
    proptest_regular_against_pulse(model, pulse, input, rank - 3)
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

/// Inverse FFT: same op with `inverse: true`. Verifies the natural
/// axes-mapping applies regardless of FFT direction.
#[test]
fn ifft_pulse_smoke_8_pulse2_size4() {
    ifft_on_non_streaming_axis(8, 2, 4).unwrap();
}

/// Rank-3 input: just (stream, fft, 2), no batch / channel dims.
#[test]
fn fft_pulse_rank3() {
    fft_inner(8, 2, 4, false, 3).unwrap();
}

/// Rank-5 input: two extra leading singleton dims before the stream
/// axis (the DPDFNet `df_op` shape is similar: `(B, C, T, F, 2)`).
#[test]
fn fft_pulse_rank5() {
    fft_inner(8, 2, 4, false, 5).unwrap();
}

/// Stacked FFT then inverse FFT on the same axis: tract should track
/// the streaming axis through both ops via the natural mapping.
#[test]
fn fft_ifft_roundtrip_pulse() {
    let mut model = TypedModel::default();
    let s = model.symbols.sym("S");
    let a = model.add_source("a", f32::fact(dims!(1, s, 4, 2))).unwrap();
    let fwd = model.wire_node("fft", Fft { axis: 2, inverse: false }, &[a]).unwrap();
    model.wire_node("ifft", Fft { axis: 2, inverse: true }, &fwd).unwrap();
    model.auto_outputs().unwrap();

    let input: ArrayD<f32> =
        ArrayD::from_shape_fn(vec![1, 8, 4, 2], |idx| (idx[1] * 8 + idx[2] * 2 + idx[3]) as f32);
    proptest_regular_against_pulse(model, 2, input, 1).unwrap();
}

/// Streaming on the FFT axis itself must be rejected: per-pulse FFT on
/// the FFT axis is meaningless (FFT needs every sample on that axis at
/// once). `Fft::axes_mapping` declares the FFT axis as input-only (it
/// does not map to any output axis), so the generic pulse fallback can
/// not track the streaming axis through it and bails. No dedicated
/// pulsifier is involved.
#[test]
fn fft_pulse_on_fft_axis_errors() {
    let mut model = TypedModel::default();
    let s = model.symbols.sym("S");
    // Streaming on axis 1, FFT also on axis 1 -> nonsense.
    let a = model.add_source("a", f32::fact(dims!(1, s, 2))).unwrap();
    model.wire_node("fft", Fft { axis: 1, inverse: false }, &[a]).unwrap();
    model.auto_outputs().unwrap();
    let err = PulsedModel::new(&model, s.clone(), &2.to_dim()).unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("could not track pulsing axis"),
        "expected the generic pulse fallback to refuse tracking the FFT \
         axis, got: {msg}"
    );
}

/// Streaming on the trailing (re, im) axis is structurally
/// impossible: `Fft::output_facts` already rejects any input whose
/// trailing axis isn't `2`, so a symbolic trailing dim trips the
/// typed-model build long before the pulsifier runs. Lock in that
/// "earlier layer rejects this" contract.
#[test]
fn fft_pulse_on_complex_axis_errors() {
    let mut model = TypedModel::default();
    let s = model.symbols.sym("S");
    // Streaming on the last axis (the complex pair).
    let a = model.add_source("a", f32::fact(dims!(1, 4, s))).unwrap();
    let err = model.wire_node("fft", Fft { axis: 1, inverse: false }, &[a]).unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("inner (last) dimension to be 2"),
        "expected the typed-model rejection of a symbolic trailing \
         axis, got: {msg}"
    );
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

    #[test]
    fn proptest_ifft_pulse(
        input_len in 1usize..16,
        pulse in 1usize..4,
        fft_size in proptest::sample::select(vec![2usize, 4, 8, 16]),
    ) {
        ifft_on_non_streaming_axis(input_len, pulse, fft_size)?
    }
}
