//! Load-time autotune probe for the Metal scheduling knobs.
//!
//! ON BY DEFAULT: at the end of `Runtime::prepare`, while the tuning
//! profile is still in its probe window (see `crate::tuning`), the Metal
//! runtime drives the freshly prepared plan with a synthetic decode-shaped
//! workload, greedily sweeps the output-invariant scheduling knobs one at a
//! time (the same candidate sets as the offline `tune_decode` sweep), and
//! adopts winners IN-MEMORY for the process. Opt out with
//! `TRACT_METAL_AUTOTUNE=0` or [`crate::tuning::set_autotune`]`(false)`;
//! opted out, nothing here runs and the profile freezes at its first read,
//! exactly the historical behavior.
//!
//! The probe is budget-capped (`TRACT_METAL_AUTOTUNE_BUDGET_MS`, default
//! 10 s) and self-limits to fit: it sizes the per-candidate iteration count
//! to the step time measured during warmup (fewer iterations rather than
//! fewer knobs), stops the sweep early keeping partial winners when the
//! budget runs out mid-way, and skips entirely (with a log) when even a
//! one-iteration sweep cannot fit. Knobs already pinned by an env var, an
//! app override or the autotune cache are not probed: better information
//! already exists, and those layers outrank probe results anyway.
//!
//! The swept knobs are read at execution time on every dispatch/commit
//! (commit cadence per `command_buffer()` acquisition, in-flight depth per
//! `commit_current`, MoE commit floor per routed dispatch), so candidates
//! can be swapped between fully quiesced runs without touching any Metal
//! object. They must never change the computed values: each candidate's
//! outputs are compared byte-for-byte against the baseline outputs and a
//! mismatch rejects the candidate loudly (it signals a tract bug, not a
//! tuning trade-off).
//!
//! Synthetic workload: inputs are built from the plan's input facts with
//! every symbolic dimension set to 1 (a decode-shaped step) and zero-filled
//! tensors (zero is a valid token id, and zero-filled KV/state tensors are
//! numerically inert). If any input cannot be confidently synthesized
//! (non-copy dtypes, exotic facts, dims that do not resolve), the probe is
//! skipped with a single warning; it never guesses into a crash.
//!
//! Winners live and die with the process: the probe never reads or writes
//! any file (some target systems have read-only disks), and at ~1-2 s on a
//! 35B-class model there is nothing worth persisting.

use std::sync::Arc;
use std::time::{Duration, Instant};

use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensorExt;

use crate::tuning::{self, MetalTuning, PinnedKnobs};

/// One probe-safe scheduling knob: profile/cache field name, candidate set
/// (mirrors ohana's offline `tune_decode`), and field accessors.
struct Knob {
    name: &'static str,
    candidates: &'static [usize],
    /// Only meaningful when the plan contains routed-MoE fast-path ops.
    moe_only: bool,
    get: fn(&MetalTuning) -> usize,
    set: fn(&mut MetalTuning, usize),
    /// Whether env/app/cache already supplies this knob (probe skips it).
    pinned: fn(&PinnedKnobs) -> bool,
}

const KNOBS: &[Knob] = &[
    Knob {
        name: "max_command_buffers_in_flight",
        candidates: &[2, 4, 8, 16],
        moe_only: false,
        get: |t| t.max_command_buffers_in_flight,
        set: |t, v| t.max_command_buffers_in_flight = v,
        pinned: |p| p.max_command_buffers_in_flight,
    },
    Knob {
        name: "commit_every_n_dispatches",
        candidates: &[0, 5, 10, 20],
        moe_only: false,
        get: |t| t.commit_every_n_dispatches,
        set: |t, v| t.commit_every_n_dispatches = v,
        pinned: |p| p.commit_every_n_dispatches,
    },
    Knob {
        name: "moe_commit_min_routes",
        candidates: &[32, 64, 128],
        moe_only: true,
        get: |t| t.moe_commit_min_routes,
        set: |t, v| t.moe_commit_min_routes = v,
        pinned: |p| p.moe_commit_min_routes,
    },
];

/// A candidate must beat the incumbent by more than this to be adopted
/// (times, so lower is better): ties and noise keep the baseline.
const NOISE_GUARD: f64 = 1.02;
/// Untimed steps before measuring (pipeline compilation, arena warmup).
const WARMUP_RUNS: usize = 3;
/// Timed steps per candidate (the median is the score), self-limited
/// downwards when the budget is tight; see [`plan_measure_runs`].
const MEASURE_RUNS_LADDER: &[usize] = &[5, 3, 2, 1];
/// Default total budget; `TRACT_METAL_AUTOTUNE_BUDGET_MS` overrides.
const DEFAULT_BUDGET_MS: u64 = 10_000;

/// Entry point, called by the Metal runtime at the end of `prepare`. On by
/// default; `TRACT_METAL_AUTOTUNE=0` / `set_autotune(false)` opts out (the
/// profile then freezes at its first read, if it has not already). Never
/// fails the session: every problem downgrades to keeping the
/// already-resolved profile.
pub(crate) fn maybe_probe(plan: &Arc<TypedSimplePlan>) {
    if tuning::is_frozen() {
        log::debug!(
            "Metal load-time autotune: tuning profile already frozen \
             (probe opted out, or an earlier prepare probed it); skipping"
        );
        return;
    }
    if !tuning::autotune_enabled() {
        // Ensure the fast frozen read path even if nothing dispatched yet
        // (the read freezes immediately when the probe is opted out).
        let _ = tuning::tuning();
        log::debug!(
            "Metal load-time autotune: disabled (TRACT_METAL_AUTOTUNE=0 or \
             set_autotune(false)); keeping the resolved profile"
        );
        return;
    }
    // Resolving with the probe enabled keeps the probe window open.
    let resolved = tuning::tuning();
    let outcome =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| probe(plan, resolved)));
    let winner = match outcome {
        Ok(Ok(profile)) => profile,
        Ok(Err(e)) => {
            log::warn!(
                "Metal load-time autotune probe abandoned ({e:?}); \
                 keeping the resolved profile"
            );
            resolved
        }
        Err(_) => {
            log::warn!(
                "Metal load-time autotune probe panicked; keeping the resolved profile"
            );
            resolved
        }
    };
    tuning::probe_freeze(winner);
    log::debug!("Metal tuning profile frozen after load-time autotune: {winner:?}");
}

fn budget() -> Duration {
    let ms = std::env::var("TRACT_METAL_AUTOTUNE_BUDGET_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(DEFAULT_BUDGET_MS);
    Duration::from_millis(ms)
}

/// The probe proper: returns the profile to freeze. Errors and skips are
/// handled by `maybe_probe` (keep the resolved profile).
fn probe(plan: &Arc<TypedSimplePlan>, resolved: MetalTuning) -> TractResult<MetalTuning> {
    let start = Instant::now();
    let budget = budget();
    let inputs = match synthesize_inputs(plan.model()) {
        Ok(inputs) => inputs,
        Err(reason) => {
            log::warn!("Metal load-time autotune probe skipped: {reason}");
            return Ok(resolved);
        }
    };
    let has_moe = plan.model().nodes().iter().any(|n| {
        n.op_is::<crate::ops::MetalRoutedQ40MatMul>()
            || n.op_is::<crate::ops::MetalRoutedQ40SwiGlu>()
    });
    // Skip knobs an env var or an app override already supplies: better
    // information exists, and those layers outrank probe results anyway.
    let pinned = tuning::pinned_knobs();
    let knobs: Vec<&Knob> = KNOBS
        .iter()
        .filter(|knob| {
            if knob.moe_only && !has_moe {
                return false;
            }
            if (knob.pinned)(&pinned) {
                log::debug!(
                    "Metal load-time autotune: not probing {} (already supplied by an \
                     env var or an app override)",
                    knob.name
                );
                return false;
            }
            true
        })
        .collect();
    if knobs.is_empty() {
        log::debug!(
            "Metal load-time autotune: every probeable knob is already supplied by an \
             env var or an app override; skipping the probe"
        );
        return Ok(resolved);
    }
    // Candidate runs to plan for (incumbent values are not re-measured).
    let candidates: usize = knobs
        .iter()
        .map(|knob| knob.candidates.iter().filter(|&&c| c != (knob.get)(&resolved)).count())
        .sum();

    let mut state = SimpleState::new(plan)?;
    let mut step = Duration::ZERO;
    for _ in 0..WARMUP_RUNS {
        let t0 = Instant::now();
        run_once(&mut state, &inputs)?;
        step = t0.elapsed();
        if start.elapsed() > budget {
            log::info!(
                "Metal load-time autotune: budget ({budget:?}) exhausted during warmup \
                 (last step {:.1} ms); keeping the resolved profile",
                step.as_secs_f64() * 1e3
            );
            return Ok(resolved);
        }
    }
    // Self-limit to the budget: size the per-candidate iteration count to
    // the measured step time (fewer iterations rather than fewer knobs).
    let remaining = budget.saturating_sub(start.elapsed());
    let Some(runs) = plan_measure_runs(step, remaining, candidates) else {
        log::warn!(
            "Metal load-time autotune: cannot fit even a 1-iteration sweep of \
             {candidates} candidates in the remaining budget (step {:.1} ms, \
             {:.0} ms left of TRACT_METAL_AUTOTUNE_BUDGET_MS={:.0}); keeping the \
             resolved profile",
            step.as_secs_f64() * 1e3,
            remaining.as_secs_f64() * 1e3,
            budget.as_secs_f64() * 1e3,
        );
        return Ok(resolved);
    };
    if runs < MEASURE_RUNS_LADDER[0] {
        log::info!(
            "Metal load-time autotune: budget-limited to {runs} iteration(s) per \
             candidate (step {:.1} ms, {candidates} candidates, {:.0} ms left)",
            step.as_secs_f64() * 1e3,
            remaining.as_secs_f64() * 1e3,
        );
    }

    // Baseline reference outputs for the output-invariance guard, with a
    // determinism pre-check: if two identical baseline runs already disagree,
    // the guard cannot attribute a mismatch to a knob and stands down (the
    // scheduling knobs are still safe to tune on timings alone).
    //
    // KNOWN NONDETERMINISM (2026-08-12, qwen3.5-35B on M4 Pro): with the
    // buffer pool enabled and a LOW in-flight depth (2), identical runs
    // intermittently differ at the byte level (varying output indices,
    // logits included; too small to flip greedy ids). Pool disabled: clean.
    // Arena disabled, pool on: still dirty. So recycled transient buffers
    // combined with early in-flight waits expose a stale-read/recycling race
    // in tract that predates this probe; the default depth 8 measured clean
    // across every check. Until that bug is fixed, a guard trip here most
    // likely means a candidate perturbed that recycling pattern; rejecting
    // the candidate is the safe response either way.
    let reference = match host_bytes(&run_once(&mut state, &inputs)?) {
        None => {
            log::warn!(
                "Metal load-time autotune: outputs are not host-comparable; \
                 probing without the output-invariance guard"
            );
            None
        }
        Some(bytes) => {
            let again = host_bytes(&run_once(&mut state, &inputs)?);
            match again.as_deref().and_then(|again| first_mismatch(&bytes, again)) {
                None => Some(bytes),
                Some(ix) => {
                    log::warn!(
                        "Metal load-time autotune: two identical baseline runs disagree \
                         (first mismatch: output #{ix}); outputs are not run-to-run \
                         deterministic on this device, probing without the \
                         output-invariance guard"
                    );
                    None
                }
            }
        }
    };

    let mut current = resolved;
    let mut current_score = measure(&mut state, &inputs, runs)?;
    log::info!(
        "Metal load-time autotune: baseline {:.3} ms/step (median of {runs}, \
         profile {current:?})",
        current_score.as_secs_f64() * 1e3
    );

    let mut adopted: Vec<(&'static str, usize)> = Vec::new();
    'sweep: for knob in &knobs {
        let incumbent = (knob.get)(&current);
        let mut best: Option<(usize, Duration)> = None;
        for &candidate in knob.candidates {
            if candidate == incumbent {
                continue;
            }
            if start.elapsed() > budget {
                log::info!(
                    "Metal load-time autotune: budget ({budget:?}) exhausted; \
                     stopping the sweep at {}={incumbent}",
                    knob.name
                );
                break 'sweep;
            }
            let mut trial = current;
            (knob.set)(&mut trial, candidate);
            tuning::probe_set(trial)?;
            // Output invariance: these knobs schedule work, they must never
            // change its result. A mismatch signals a tract bug (see the
            // known buffer-pool/in-flight nondeterminism note above); either
            // way the candidate must not be adopted.
            let out = host_bytes(&run_once(&mut state, &inputs)?);
            if let (Some(reference), Some(out)) = (&reference, &out) {
                if let Some(ix) = first_mismatch(reference, out) {
                    log::error!(
                        "Metal load-time autotune: {}={candidate} CHANGED THE OUTPUTS \
                         (first mismatch: output #{ix}). Scheduling knobs must be \
                         output-invariant: this points at a scheduling-sensitive \
                         numerics bug in tract (known suspect: transient buffer \
                         recycling under low in-flight depth), report it. \
                         Candidate rejected.",
                        knob.name
                    );
                    continue;
                }
            }
            let score = measure(&mut state, &inputs, runs)?;
            log::info!(
                "Metal load-time autotune: {}={candidate}: {:.3} ms/step \
                 (incumbent {incumbent}: {:.3} ms/step)",
                knob.name,
                score.as_secs_f64() * 1e3,
                current_score.as_secs_f64() * 1e3
            );
            if best.is_none_or(|(_, s)| score < s) {
                best = Some((candidate, score));
            }
        }
        // Put the incumbent profile back before deciding.
        tuning::probe_set(current)?;
        match best {
            Some((candidate, score))
                if score.as_secs_f64() * NOISE_GUARD < current_score.as_secs_f64() =>
            {
                (knob.set)(&mut current, candidate);
                tuning::probe_set(current)?;
                log::info!(
                    "Metal load-time autotune: adopted {}={candidate} in-memory \
                     ({:.3} vs {:.3} ms/step, {:+.1}%)",
                    knob.name,
                    score.as_secs_f64() * 1e3,
                    current_score.as_secs_f64() * 1e3,
                    (score.as_secs_f64() / current_score.as_secs_f64() - 1.0) * 100.0
                );
                current_score = score;
                adopted.push((knob.name, candidate));
            }
            _ => log::info!(
                "Metal load-time autotune: kept {}={incumbent} \
                 (no candidate beat it by >{:.0}%)",
                knob.name,
                (NOISE_GUARD - 1.0) * 100.0
            ),
        }
    }

    if adopted.is_empty() {
        log::info!(
            "Metal load-time autotune: no knob beat the resolved profile; \
             nothing adopted ({:.1} s)",
            start.elapsed().as_secs_f64()
        );
        return Ok(resolved);
    }

    // Final combined invariance check of the adopted profile (each winner
    // was individually checked, greedily chained; this recheck is cheap).
    tuning::probe_set(current)?;
    let out = host_bytes(&run_once(&mut state, &inputs)?);
    if let (Some(reference), Some(out)) = (&reference, &out) {
        if let Some(ix) = first_mismatch(reference, out) {
            log::error!(
                "Metal load-time autotune: the adopted combination {adopted:?} CHANGED \
                 THE OUTPUTS (first mismatch: output #{ix}). This is a tract bug, \
                 report it. Keeping the resolved profile."
            );
            return Ok(resolved);
        }
    }
    log::info!(
        "Metal load-time autotune: adopted {adopted:?} in-memory for this process \
         ({:.1} s probe)",
        start.elapsed().as_secs_f64()
    );
    Ok(current)
}

/// One synchronized step: run the plan and wait for the device to go idle,
/// so wall time covers the full step and knob swaps see a quiesced stream.
fn run_once(state: &mut TypedSimpleState, inputs: &TVec<TValue>) -> TractResult<TVec<TValue>> {
    let outputs = state.run(inputs.clone())?;
    tract_gpu::device::get_context()?.synchronize()?;
    Ok(outputs)
}

/// Median step time over `runs` synchronized runs.
fn measure(
    state: &mut TypedSimpleState,
    inputs: &TVec<TValue>,
    runs: usize,
) -> TractResult<Duration> {
    let mut times = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t0 = Instant::now();
        run_once(state, inputs)?;
        times.push(t0.elapsed());
    }
    times.sort();
    Ok(times[times.len() / 2])
}

/// The largest per-candidate iteration count on the ladder whose full sweep
/// (baseline measurement, determinism pre-check, one invariance run plus
/// the iterations per candidate, final combined check) fits the remaining
/// budget with ~10% headroom; `None` when even one iteration per candidate
/// does not fit. Preferring fewer iterations over dropping knobs keeps the
/// sweep's coverage; the per-candidate budget check still stops early if
/// the estimate proves optimistic.
fn plan_measure_runs(step: Duration, remaining: Duration, candidates: usize) -> Option<usize> {
    let total_runs =
        |runs: usize| 2 + runs + candidates * (1 + runs) + 1;
    MEASURE_RUNS_LADDER
        .iter()
        .copied()
        .find(|&runs| step.saturating_mul(total_runs(runs) as u32) <= remaining.mul_f64(0.9))
}

/// Synthesize one decode-shaped input set from the plan's input facts:
/// every symbolic dimension resolves to 1, every tensor is zero-filled.
/// `Err(reason)` when any input cannot be confidently synthesized.
fn synthesize_inputs(model: &TypedModel) -> Result<TVec<TValue>, String> {
    let mut values = SymbolValues::default();
    for symbol in model.symbols.all_symbols() {
        values.set(&symbol, 1);
    }
    let inputs = model.input_outlets().map_err(|e| format!("no input outlets: {e}"))?;
    let mut synthesized = tvec![];
    for (ix, outlet) in inputs.iter().enumerate() {
        let fact = model
            .outlet_fact(*outlet)
            .map_err(|e| format!("no fact for input #{ix}: {e}"))?;
        if fact.exotic_fact.is_some() || !fact.datum_type.is_copy() {
            return Err(format!(
                "input #{ix} has a non-synthesizable type ({:?})",
                fact.datum_type
            ));
        }
        let mut shape = TVec::with_capacity(fact.shape.rank());
        for (axis, dim) in fact.shape.iter().enumerate() {
            match dim.eval_to_i64(&values) {
                Ok(d) if d >= 0 => shape.push(d as usize),
                _ => {
                    return Err(format!(
                        "input #{ix} axis {axis} dimension `{dim}` does not resolve \
                         with all symbols set to 1"
                    ));
                }
            }
        }
        let tensor = Tensor::zero_dt(fact.datum_type, &shape)
            .map_err(|e| format!("cannot zero-fill input #{ix}: {e}"))?;
        synthesized.push(tensor.into_tvalue());
    }
    Ok(synthesized)
}

/// Index of the first differing output between two byte captures, `None`
/// when they are identical.
fn first_mismatch(a: &[Vec<u8>], b: &[Vec<u8>]) -> Option<usize> {
    if a.len() != b.len() {
        return Some(a.len().min(b.len()));
    }
    a.iter().zip(b.iter()).position(|(a, b)| a != b)
}

/// The outputs as host byte strings for the invariance comparison, `None`
/// when any output cannot be compared (non-copy host dtype, or a device
/// tensor that cannot be read back).
fn host_bytes(outputs: &TVec<TValue>) -> Option<Vec<Vec<u8>>> {
    let mut all = Vec::with_capacity(outputs.len());
    for output in outputs {
        if let Some(device) = output.as_device_tensor() {
            match device.to_host() {
                Ok(host) if host.datum_type().is_copy() => all.push(host.as_bytes().to_vec()),
                _ => return None,
            }
        } else if output.datum_type().is_copy() {
            all.push(output.as_bytes().to_vec());
        } else {
            return None;
        }
    }
    Some(all)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Symbolic dims resolve to 1 (decode shape), tensors are zero-filled.
    #[test]
    fn synthesize_decode_shaped_zero_inputs() {
        let mut model = TypedModel::default();
        let s = model.symbols.sym("S");
        let p = model.symbols.sym("P");
        let ids = model
            .add_source("input_ids", i64::fact(&[s.to_dim()]))
            .unwrap();
        let kv = model
            .add_source("kv", f32::fact(&[1.into(), 4.into(), p.to_dim(), 8.into()]))
            .unwrap();
        model.select_output_outlets(&[ids, kv]).unwrap();
        let inputs = synthesize_inputs(&model).unwrap();
        assert_eq!(inputs.len(), 2);
        assert_eq!(inputs[0].shape(), &[1]);
        assert_eq!(inputs[0].datum_type(), i64::datum_type());
        assert_eq!(inputs[0].try_as_plain().unwrap().as_slice::<i64>().unwrap(), &[0]);
        assert_eq!(inputs[1].shape(), &[1, 4, 1, 8]);
        assert!(
            inputs[1]
                .try_as_plain()
                .unwrap()
                .as_slice::<f32>()
                .unwrap()
                .iter()
                .all(|&v| v == 0.0)
        );
    }

    /// A non-synthesizable input (non-copy dtype) declines the probe with a
    /// reason instead of guessing.
    #[test]
    fn synthesize_declines_non_copy_dtypes() {
        let mut model = TypedModel::default();
        let x = model
            .add_source("x", TypedFact::dt_shape(DatumType::TDim, [1usize, 2]))
            .unwrap();
        model.select_output_outlets(&[x]).unwrap();
        let err = synthesize_inputs(&model).unwrap_err();
        assert!(err.contains("non-synthesizable"), "{err}");
    }

    /// Output-invariance comparison: byte differences are detected, non-copy
    /// outputs make the comparison decline (None) rather than lie.
    #[test]
    fn host_bytes_detects_differences_and_declines_non_copy() {
        let a = tvec![tensor1(&[1f32, 2., 3.]).into_tvalue()];
        let b = tvec![tensor1(&[1f32, 2., 4.]).into_tvalue()];
        assert_eq!(host_bytes(&a), host_bytes(&a));
        assert_ne!(host_bytes(&a), host_bytes(&b));
        let opaque = tvec![tensor1(&[TDim::from(1)]).into_tvalue()];
        assert!(host_bytes(&opaque).is_none());
    }

    /// The budget planner prefers fewer iterations over dropping knobs, and
    /// declines (None) when even one iteration per candidate cannot fit.
    #[test]
    fn budget_planner_self_limits() {
        let ms = Duration::from_millis;
        // Plenty of budget: full 5 iterations.
        assert_eq!(plan_measure_runs(ms(10), ms(10_000), 8), Some(5));
        // Tight budget: degrade iterations, keep every candidate.
        // 8 candidates: total runs are 56 (n=5), 38 (n=3), 29 (n=2), 20 (n=1).
        assert_eq!(plan_measure_runs(ms(10), ms(500), 8), Some(3));
        assert_eq!(plan_measure_runs(ms(10), ms(250), 8), Some(1));
        // Even one iteration per candidate does not fit: decline.
        assert_eq!(plan_measure_runs(ms(10), ms(100), 8), None);
        assert_eq!(plan_measure_runs(ms(1000), ms(1000), 3), None);
    }
}
