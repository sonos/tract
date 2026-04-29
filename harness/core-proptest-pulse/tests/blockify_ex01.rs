//! End-to-end test for the Blockify rewrite on the
//! `harness/pulse-multi-axis/ex01-block-diag-reduce` synthetic.
//!
//! Loads the NNEF graph, runs batch and pulsified, and compares
//! numerically.

use tract_core::ndarray::{Array, Array2, Axis};
use tract_core::prelude::*;
use tract_pulse::internal::*;

fn ex_path(name: &str) -> std::path::PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    std::path::Path::new(&manifest)
        .join("../../harness/pulse-multi-axis")
        .join(name)
        .canonicalize()
        .unwrap()
}

fn ex01_path() -> std::path::PathBuf {
    ex_path("ex01-block-diag-reduce")
}

#[test]
fn ex01_block_diag_reduce_blockified_pulse_matches_batch() {
    let nnef = tract_nnef::nnef().with_tract_core();
    let model = nnef.model_for_path(ex01_path()).unwrap().into_decluttered().unwrap();
    let t = model.symbols.sym("T");

    // Reference inputs.  P=2, S=3, T=6, D=4.
    let a = Array2::<f32>::from_shape_fn((6, 4), |(i, j)| (i * 4 + j) as f32 * 0.1);
    let b = Array2::<f32>::from_shape_fn((6, 4), |(i, j)| ((i * 4 + j) as f32 - 5.0) * 0.1);

    // Concrete batch run: T=6.
    let subs = std::collections::HashMap::from([(t.clone(), TDim::Val(6))]);
    let concrete = model.clone().substitute_symbols(&subs).unwrap();
    let runnable = concrete.into_runnable().unwrap();
    let batch_out = runnable
        .run(tvec!(a.clone().into_dyn().into_tvalue(), b.clone().into_dyn().into_tvalue()))
        .unwrap();
    let batch_arr = batch_out[0].to_plain_array_view::<f32>().unwrap().to_owned();
    assert_eq!(batch_arr.shape(), &[6]);

    // Pulsify with --pulse T=2.  Blockify fires inside PulsedModel::new.
    let pulsed = PulsedModel::new(&model, t, &2.to_dim()).unwrap();
    let plan = SimplePlan::new(pulsed.into_typed().unwrap()).unwrap();
    let mut state = SimpleState::new(&plan).unwrap();

    let mut got: Vec<f32> = Vec::new();
    for chunk in 0..3 {
        let a_slice = a.slice_axis(Axis(0), (chunk * 2..(chunk + 1) * 2).into()).to_owned();
        let b_slice = b.slice_axis(Axis(0), (chunk * 2..(chunk + 1) * 2).into()).to_owned();
        let out = state
            .run(tvec!(a_slice.into_dyn().into_tvalue(), b_slice.into_dyn().into_tvalue()))
            .unwrap();
        got.extend(out[0].to_plain_array_view::<f32>().unwrap().iter());
    }

    let pulsed_arr = Array::from(got);
    assert_eq!(pulsed_arr.shape(), &[6]);
    for (i, (b_val, p_val)) in batch_arr.iter().zip(pulsed_arr.iter()).enumerate() {
        assert!((b_val - p_val).abs() < 1e-4, "mismatch at {i}: batch={b_val} pulsed={p_val}");
    }
}

/// Probe the failure mode for ex02-block-diag-bilinear: where exactly
/// does the pulse pipeline give up, and is it an Err or a panic?
///
/// Currently expected to fail somewhere — this test asserts where so we
/// notice if the boundary moves (better or worse) as Blockify evolves.
#[test]
fn ex02_block_diag_bilinear_currently_fails_at_run_time_with_recoverable_error() {
    let nnef = tract_nnef::nnef().with_tract_core();
    let model = nnef
        .model_for_path(ex_path("ex02-block-diag-bilinear"))
        .unwrap()
        .into_decluttered()
        .unwrap();
    let t = model.symbols.sym("T");

    // Blockify (no-op for ex02 today) + pulsifier — both succeed even
    // though the resulting graph has unresolved symbols ("range").
    // Pulsification's inability to refuse here is the silent-garbage
    // failure mode that motivates strict-gating.
    let pulsed = PulsedModel::new(&model, t, &2.to_dim()).expect("PulsedModel::new succeeds");
    let typed = pulsed.into_typed().expect("into_typed succeeds");
    let plan = SimplePlan::new(typed).expect("SimplePlan::new succeeds with unresolved symbols");

    // The error surfaces only when we try to actually run the plan with
    // concrete inputs.  It's a clean Result::Err — recoverable, no panic.
    let mut state = SimpleState::new(&plan).expect("SimpleState::new succeeds");
    let a = tract_core::ndarray::Array2::<f32>::zeros((2, 4));
    let b = tract_core::ndarray::Array2::<f32>::zeros((2, 4));
    let c = tract_core::ndarray::Array2::<f32>::zeros((2, 4));
    let res = state.run(tvec!(
        a.into_dyn().into_tvalue(),
        b.into_dyn().into_tvalue(),
        c.into_dyn().into_tvalue()
    ));
    let err = res.expect_err("ex02 plan-run should fail with a recoverable Err");
    let chain = format!("{err:?}\n{:?}", err.root_cause());
    assert!(
        chain.contains("symbol") || chain.contains("range") || chain.contains("Undetermined"),
        "unexpected error: {chain}"
    );
}
