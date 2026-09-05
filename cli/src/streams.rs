use std::time::Instant;

use tract_core::lanes::LanedRunnable;
use tract_hir::internal::*;
use tract_libcli::tensor::get_or_make_inputs;

use crate::params::Parameters;

/// Run the model on several concurrent streams, and check every one of them
/// gets what it gets alone.
///
/// One thread and one state per stream -- one lane each when the runtime
/// autobatches (`--autobatch-sessions`), one state of its own when it does not
/// -- feeding one row
/// per turn as fast as it can. So the seats stagger themselves: occupancy is
/// whatever the worker finds queued, and a piece of state two streams share
/// shows up as a diff against the solo run. Stream `k` feeds the input turns
/// rotated by `k`, so no two seats of a turn carry the same values -- a state
/// shared by two streams sitting at the same position in the same signal is
/// invisible.
///
/// The stream-length symbol is never bound: a turn is symbol-homogeneous, so
/// end of stream belongs to the caller. Feed whole pulses.
///
/// Returns the first stream's outputs, per output and per turn, so that the
/// assertions on a plain run apply to it unchanged.
pub fn run(
    params: &Parameters,
    sub_matches: &clap::ArgMatches,
    streams: usize,
) -> TractResult<TVec<Vec<TValue>>> {
    ensure!(
        !sub_matches.get_flag("steps") && !sub_matches.contains_id("save-steps"),
        "Streams are served through their own states, which the per-node hooks of \
         --steps and --save-steps can not reach"
    );
    let runnable = params.req_runnable()?;
    let laned = runnable.downcast_ref::<LanedRunnable>();
    if let Some(laned) = laned {
        ensure!(
            streams <= laned.max_lanes(),
            "{streams} streams over {} autobatch sessions: a stream holds one for its whole life",
            laned.max_lanes()
        );
    } else {
        ensure!(
            !sub_matches.contains_id("assert-occupancy"),
            "Occupancy is what an autobatched turn carries: --assert-occupancy wants --autobatch-sessions"
        );
    }
    let solo = laned.map(|laned| laned.inner()).unwrap_or(&runnable);

    let mut run_params = crate::tensor::run_params_from_subcommand(params, sub_matches)?;
    if let Some(laned) = laned {
        run_params.symbols.set(laned.batch_symbol(), 1);
    }
    let sources = get_or_make_inputs(&params.tract_model, &run_params)?.sources;
    let turns: usize = sub_matches
        .get_one::<String>("turns")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(sources.len());
    ensure!(
        sources.len() > 1 || streams == 1,
        "The input covers one turn, so every stream would feed the same values: \
         give a longer input"
    );
    ensure!(turns > 0, "A run needs a turn at least");

    let start = Instant::now();
    let got: Vec<Vec<TVec<TValue>>> = std::thread::scope(|scope| {
        let running: Vec<_> = (0..streams)
            .map(|stream| {
                let runnable = runnable.clone();
                let sources = &sources;
                scope.spawn(move || -> TractResult<Vec<TVec<TValue>>> {
                    let mut state = runnable.spawn()?;
                    (0..turns)
                        .map(|turn| state.run(sources[(turn + stream) % sources.len()].clone()))
                        .collect()
                })
            })
            .collect();
        running.into_iter().map(|stream| stream.join().unwrap()).collect::<Vec<_>>()
    })
    .into_iter()
    .collect::<TractResult<_>>()?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    println!(
        "{streams} streams x {turns} turns in {elapsed:.0} ms, {:.3} ms/turn/stream",
        elapsed / (streams * turns) as f64
    );
    if let Some(laned) = laned {
        let (worked, seats) = laned.turns_and_seats();
        let occupancy = seats as f64 / worked as f64;
        println!("{worked} turns, {seats} seats, mean occupancy {occupancy:.2}");
        if let Some(least) = sub_matches.get_one::<String>("assert-occupancy") {
            let least: f64 = least.parse()?;
            ensure!(
                occupancy >= least,
                "Mean occupancy {occupancy:.2} under the {least} asserted: the turns carried \
                 one stream at a time, so nothing of the batch axis was exercised"
            );
        }
    }

    let approx = params.assertions.approximation;
    // Per turn, over every stream: the worst absolute diff and the largest
    // reference magnitude. Reassociation noise holds a ratio; a state two
    // streams share compounds turn over turn.
    let mut per_turn = vec![(0f32, 0f32); turns];
    for (stream, got) in got.iter().enumerate() {
        let mut solo = solo.spawn()?;
        for (turn, got) in got.iter().enumerate() {
            let want = solo.run(sources[(turn + stream) % sources.len()].clone())?;
            for (ix, (got, want)) in got.iter().zip(want.iter()).enumerate() {
                got.close_enough(want, approx).with_context(|| {
                    format!("Stream {stream}, turn {turn}, output {ix} against running alone")
                })?;
                let (diff, scale) = magnitudes(got, want)?;
                per_turn[turn].0 = per_turn[turn].0.max(diff);
                per_turn[turn].1 = per_turn[turn].1.max(scale);
            }
        }
    }
    println!(
        "worst diff against running alone: {:e}",
        per_turn.iter().map(|(diff, _)| *diff).fold(0f32, f32::max)
    );
    if sub_matches.get_flag("per-turn-diff") {
        println!("\nturn  worst abs    |want|max   relative");
        for (turn, (diff, scale)) in per_turn.iter().enumerate() {
            println!("{turn:4}  {diff:9.3e}  {scale:9.3e}  {:9.3e}", diff / scale);
        }
    }

    if params.assertions.assert_outputs {
        ensure!(
            turns == sources.len(),
            "An output bundle covers the whole stream, so --turns can not change it"
        );
    }
    let outputs = params.tract_model.output_outlets().len();
    Ok((0..outputs).map(|ix| got[0].iter().map(|turn| turn[ix].clone()).collect()).collect())
}

/// The worst absolute difference between two float tensors and the largest
/// magnitude of the reference, or zeroes for anything else -- what the
/// approximation check already covers exactly.
fn magnitudes(got: &Tensor, want: &Tensor) -> TractResult<(f32, f32)> {
    if !want.datum_type().is_float() {
        return Ok((0., 0.));
    }
    let want = want.cast_to::<f32>()?;
    let want = want.to_plain_array_view::<f32>()?;
    let got = got.cast_to::<f32>()?;
    let diff = got
        .to_plain_array_view::<f32>()?
        .iter()
        .zip(want.iter())
        .map(|(got, want)| (got - want).abs())
        .fold(0f32, f32::max);
    Ok((diff, want.iter().map(|want| want.abs()).fold(0f32, f32::max)))
}
