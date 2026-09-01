use infra::{TestResult, TestSuite};
use tract_core::internal::*;
use tract_core::ndarray::{ArrayD, Axis};
use tract_core::num_traits::Zero;
use tract_core::runtime::Runtime;
use tract_pulse::internal::PulsedModel;
use tract_pulse::model::PulsedModelExt;

pub mod conv_plus_conv;
pub mod deconv;
pub mod delay_plus_downsample;
pub mod delay_plus_pool;
pub mod einsum;
pub mod fft;
pub mod lanes;
pub mod pad;
pub mod pad_plus_conv;
pub mod slice;
pub mod stft;

pub fn suite() -> TractResult<TestSuite> {
    let mut suite: TestSuite = Default::default();
    suite.add("deconv", deconv::suite()?);
    suite.add("delay_plus_downsample", delay_plus_downsample::suite()?);
    suite.add("delay_plus_pool", delay_plus_pool::suite()?);
    suite.add("conv_plus_conv", conv_plus_conv::suite()?);
    suite.add("einsum", einsum::suite()?);
    suite.add("fft", fft::suite()?);
    suite.add("pad_plus_conv", pad_plus_conv::suite()?);
    suite.add("lanes", lanes::suite()?);
    suite.add("pad", pad::suite()?);
    suite.add("slice", slice::suite()?);
    suite.add("stft", stft::suite()?);
    Ok(suite)
}

/// Runs `model` with its streaming symbol concretized to the input length, then
/// pulsified at `pulse`, and checks the streamed output against the reference.
/// Both runs go through `runtime`, so what is compared is batch against pulsed,
/// not one backend against another.
pub fn pulse_and_compare(
    runtime: &dyn Runtime,
    approx: Approximation,
    model: TypedModel,
    pulse: usize,
    input: ArrayD<f32>,
    axis: usize,
) -> TestResult {
    let len = input.shape()[axis];
    let model = model.into_decluttered()?;
    let s = model.symbols.sym("S");

    let subs = std::collections::HashMap::from([(s.clone(), TDim::Val(len as i64))]);
    let concrete = model.clone().set_symbols(&subs)?;
    // A wire the concrete length empties has nothing to compare: the reference
    // run would be all shape and no value.
    if concrete.nodes.iter().any(|n| n.outputs.iter().any(|o| o.fact.shape.volume().is_zero())) {
        return Ok(());
    }
    let reference =
        runtime.prepare(concrete)?.run(tvec!(input.clone().into_tensor().into_tvalue()))?;

    let pulsed = PulsedModel::new(&model, s.clone(), &pulse.to_dim())?;
    let output_fact = pulsed.output_fact(0)?.clone();
    let stream = output_fact.stream.clone().context("Pulsed output is not streaming")?;
    let full = SymbolValues::default().with(&s, len as i64);
    ensure!(
        stream.dim.eval(&full) == reference[0].shape()[stream.axis].to_dim(),
        "Pulsed output declares {} frames, reference has {}",
        stream.dim.eval(&full),
        reference[0].shape()[stream.axis]
    );

    let mut empty_output_shape = output_fact.shape.clone();
    empty_output_shape.set(stream.axis, 0.to_dim());
    let empty_output_shape: TVec<usize> =
        empty_output_shape.iter().map(|d| Ok(d.to_usize()?)).collect::<TractResult<_>>()?;

    let mut state = runtime.prepare(pulsed.into_typed()?)?.spawn()?;
    let mut got: ArrayD<f32> = ArrayD::zeros(&*empty_output_shape);
    let mut output_len = None;
    let mut written = 0;
    loop {
        let live = pulse.min(len.saturating_sub(written));
        let mut chunk = input.slice_axis(Axis(axis), (written..written + live).into()).to_owned();
        written += live;
        if live < pulse {
            // The last pulse is padded to full width, so only the symbol can
            // tell the plan where the stream actually ended.
            let mut filler_shape = input.shape().to_vec();
            filler_shape[axis] = pulse - live;
            chunk = tract_core::ndarray::concatenate(
                Axis(axis),
                &[chunk.view(), ArrayD::from_elem(filler_shape, f32::NAN).view()],
            )?;
            state.resolve_symbol(&s, written as i64)?;
            let ends_at = SymbolValues::default().with(&s, written as i64);
            output_len = stream.dim.eval(&ends_at).to_isize().ok().map(|n| n.max(0) as usize);
        }
        let mut outputs = state.run(tvec!(chunk.into_tensor().into_tvalue()))?;
        got = tract_core::ndarray::concatenate(
            Axis(stream.axis),
            &[got.view(), outputs.remove(0).to_plain_array_view::<f32>()?],
        )?;
        if let Some(output_len) = output_len
            && got.shape()[stream.axis] >= output_len + stream.delay
        {
            break;
        }
    }

    let output_len = output_len.context("Stream never ended")?;
    let pulsed_output = got
        .slice_axis(Axis(stream.axis), (stream.delay..stream.delay + output_len).into())
        .to_owned()
        .into_tensor();
    pulsed_output
        .close_enough(&reference[0], approx)
        .with_context(|| format!("pulsed {pulsed_output:?} against reference {:?}", reference[0]))
}

/// A strategy for `len` small float values, the input material every family
/// feeds its streaming model.
pub fn values(
    len: impl proptest::strategy::Strategy<Value = usize>,
) -> impl proptest::strategy::Strategy<Value = Vec<f32>> {
    use proptest::strategy::Strategy;
    len.prop_flat_map(|l| proptest::collection::vec(-5..5, l..=l))
        .prop_map(|v| v.into_iter().map(|f| f as f32).collect())
}
