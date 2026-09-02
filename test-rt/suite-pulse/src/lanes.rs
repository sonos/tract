use infra::{Test, TestResult, TestSuite};
use tract_core::internal::*;
use tract_core::lanes::LanedRunnable;
use tract_core::ndarray::{ArrayD, Axis, arr3};
use tract_core::ops::array::{Pad, PadMode};
use tract_core::ops::cnn::{Conv, KernelFormat, PaddingSpec, PoolSpec};
use tract_core::ops::nn::DataFormat;
use tract_core::runtime::{RunOptions, Runnable, Runtime};
use tract_pulse::internal::PulsedModel;
use tract_pulse::model::PulsedModelExt;

/// A turn seating several streams must hand each of them the output it gets
/// running alone. `turns` lists the (lane, stream) pairs seated at each turn,
/// staggered so that a lane joins late and sits turns out, and `recycle` is the
/// turn at which lane 0 is reset and handed to a new stream.
#[derive(Clone, Debug)]
pub struct LanesProblem {
    pub pulse: usize,
    pub max_lanes: usize,
    pub pad_before: usize,
    pub pad_mode: PadMode,
    pub turns: Vec<Vec<(usize, usize)>>,
    pub recycle: Option<usize>,
}

/// Stream `stream`'s `turn`th pulse, so every stream carries a signal of its own.
fn pulse_of(stream: usize, turn: usize, pulse: usize) -> ArrayD<f32> {
    ArrayD::from_shape_fn(vec![1, 1, pulse], |ix| (100 * stream + 10 * turn + ix[2]) as f32)
}

fn stack(pulses: &[ArrayD<f32>]) -> TractResult<TValue> {
    let views: TVec<_> = pulses.iter().map(|p| p.view()).collect();
    Ok(tract_core::ndarray::concatenate(Axis(0), &views)?.into_tensor().into_tvalue())
}

/// Pad plus convolution over `[B, 1, S]`: pulsified, the padding becomes a
/// `PulsePad` and the kernel window a `Delay`, both carrying the batch axis the
/// lanes are addressed along.
fn pulsed(pulse: usize, pad_before: usize, pad_mode: &PadMode) -> TractResult<TypedModel> {
    let mut model = TypedModel::default();
    let s = model.symbols.sym("S");
    let b = model.symbols.sym("B");
    let mut wire = model.add_source("a", f32::fact(dims!(b, 1, s)))?;
    wire = model.wire_node(
        "pad",
        Pad::new(vec![(0, 0), (0, 0), (pad_before, 0)], pad_mode.clone()),
        &[wire],
    )?[0];
    let kernel = model.add_const("kernel", arr3(&[[[1f32, 2., 3.]]]))?;
    let bias = model.add_const("bias", tensor0(0f32))?;
    let conv = model.wire_node(
        "conv",
        Conv {
            pool_spec: PoolSpec::new(
                DataFormat::NCHW,
                tvec!(3),
                PaddingSpec::Valid,
                None,
                None,
                1,
                1,
            ),
            kernel_fmt: KernelFormat::OIHW,
            group: 1,
            q_params: None,
        },
        &[wire, kernel, bias],
    )?;
    model.select_output_outlets(&conv)?;
    PulsedModel::new(&model.into_decluttered()?, s, &pulse.to_dim())?.into_typed()
}

/// One stream on a state of its own: the reference every laned turn is compared
/// to.
fn solo(
    runnable: &dyn Runnable,
    pulse: usize,
    stream: usize,
    turns: usize,
) -> TractResult<Vec<Tensor>> {
    let mut state = runnable.spawn()?;
    (0..turns)
        .map(|turn| {
            let input = stack(&[pulse_of(stream, turn, pulse)])?;
            Ok(state.run(tvec!(input))?.remove(0).into_tensor())
        })
        .collect()
}

/// How wide a turn gets: the batch axis the lanes sit on stays symbolic, so an
/// arena has to be told before it can size its buffers.
fn options(model: &TypedModel, max_lanes: usize) -> RunOptions {
    let batch = model.symbols.sym("B");
    RunOptions {
        memory_sizing_hints: Some(SymbolValues::default().with(&batch, max_lanes as i64)),
        ..RunOptions::default()
    }
}

impl Test for LanesProblem {
    fn run_with_approx(
        &self,
        _id: &str,
        runtime: &dyn Runtime,
        approx: Approximation,
    ) -> TestResult {
        let model = pulsed(self.pulse, self.pad_before, &self.pad_mode)?;
        let options = options(&model, self.max_lanes);
        let runnable = runtime.prepare_with_options(model, &options)?;
        let streams = self.turns.iter().flatten().map(|(_, stream)| stream + 1).max().unwrap_or(0);
        let mut state = runnable.spawn()?;
        let mut turns_of_stream = vec![0usize; streams];
        let mut got: Vec<Vec<Tensor>> = vec![vec![]; streams];
        for (turn, seats) in self.turns.iter().enumerate() {
            if self.recycle == Some(turn) {
                state.reset_lanes(&[LaneId(0)])?;
            }
            let seating =
                Seating::new(self.max_lanes, seats.iter().map(|(lane, _)| LaneId(*lane)))?;
            state.seat(seating)?;
            let pulses: Vec<ArrayD<f32>> = seats
                .iter()
                .map(|(_, stream)| pulse_of(*stream, turns_of_stream[*stream], self.pulse))
                .collect();
            let output = state.run(tvec!(stack(&pulses)?))?.remove(0).into_tensor();
            for (seat, (_, stream)) in seats.iter().enumerate() {
                got[*stream].push(output.slice(0, seat, seat + 1)?);
                turns_of_stream[*stream] += 1;
            }
        }

        for (stream, got) in got.iter().enumerate() {
            let expected = solo(&*runnable, self.pulse, stream, got.len())?;
            for (turn, (got, expected)) in got.iter().zip(expected.iter()).enumerate() {
                got.close_enough(expected, approx)
                    .with_context(|| format!("stream {stream} turn {turn}"))?;
            }
        }
        Ok(())
    }
}

/// The laned runtime serving one stream per thread. Each stream takes a lane of
/// its own and feeds one pulse per turn as fast as it can, so occupancy is
/// whatever the worker finds queued and the seats stagger themselves.
#[derive(Clone, Debug)]
pub struct LanedProblem {
    pub pulse: usize,
    pub pad_before: usize,
    pub pad_mode: PadMode,
    pub streams: usize,
    pub turns: usize,
}

impl Test for LanedProblem {
    fn run_with_approx(
        &self,
        _id: &str,
        runtime: &dyn Runtime,
        approx: Approximation,
    ) -> TestResult {
        let model = pulsed(self.pulse, self.pad_before, &self.pad_mode)?;
        let options = options(&model, self.streams);
        let laned = LanedRunnable::wrap(
            runtime.prepare_with_options(model.clone(), &options)?,
            self.streams,
        )?;
        let got: Vec<Vec<Tensor>> = std::thread::scope(|scope| {
            let streams: Vec<_> = (0..self.streams)
                .map(|stream| {
                    let laned = laned.clone();
                    scope.spawn(move || -> TractResult<Vec<Tensor>> {
                        let mut handle = laned.spawn()?;
                        (0..self.turns)
                            .map(|turn| {
                                let input = stack(&[pulse_of(stream, turn, self.pulse)])?;
                                Ok(handle.run(tvec!(input))?.remove(0).into_tensor())
                            })
                            .collect()
                    })
                })
                .collect();
            streams.into_iter().map(|stream| stream.join().unwrap()).collect::<Vec<_>>()
        })
        .into_iter()
        .collect::<TractResult<_>>()?;

        let runnable = runtime.prepare_with_options(model, &options)?;
        for (stream, got) in got.iter().enumerate() {
            let expected = solo(&*runnable, self.pulse, stream, got.len())?;
            for (turn, (got, expected)) in got.iter().zip(expected.iter()).enumerate() {
                got.close_enough(expected, approx)
                    .with_context(|| format!("stream {stream} turn {turn}"))?;
            }
        }
        Ok(())
    }
}

pub fn suite() -> TractResult<TestSuite> {
    let mut suite = TestSuite::default();
    let staggered = |pad_before, pad_mode| LanesProblem {
        pulse: 2,
        max_lanes: 3,
        pad_before,
        pad_mode,
        // (lane, stream) pairs seated at each turn.
        turns: vec![
            vec![(0, 0)],
            vec![(0, 0), (1, 1)],
            vec![(1, 1)],
            vec![(0, 0), (1, 1), (2, 2)],
            vec![(2, 2)],
            vec![(0, 3), (2, 2)],
            vec![(0, 3), (1, 1), (2, 2)],
        ],
        recycle: Some(5),
    };
    suite.add_test(
        "staggered_seats_constant_pad",
        staggered(2, PadMode::Constant(tensor0(9999f32).into())),
    );
    suite.add_test("staggered_seats_edge_pad", staggered(1, PadMode::Edge));
    suite.add_test(
        "laned_runtime",
        LanedProblem {
            pulse: 2,
            pad_before: 2,
            pad_mode: PadMode::Constant(tensor0(9999f32).into()),
            streams: 4,
            turns: 16,
        },
    );
    Ok(suite)
}
