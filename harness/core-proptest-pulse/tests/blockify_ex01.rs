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

#[test]
fn ex02_block_diag_bilinear_blockified_pulse_matches_batch() {
    let nnef = tract_nnef::nnef().with_tract_core();
    let model = nnef
        .model_for_path(ex_path("ex02-block-diag-bilinear"))
        .unwrap()
        .into_decluttered()
        .unwrap();
    let t = model.symbols.sym("T");

    // Reference inputs.  P=2, S=3, T=6, D=4.  Three streams a, b, c.
    let a = Array2::<f32>::from_shape_fn((6, 4), |(i, j)| (i * 4 + j) as f32 * 0.1);
    let b = Array2::<f32>::from_shape_fn((6, 4), |(i, j)| ((i * 4 + j) as f32 - 5.0) * 0.1);
    let c = Array2::<f32>::from_shape_fn((6, 4), |(i, j)| ((i * 4 + j) as f32 + 3.0) * 0.05);

    let subs = std::collections::HashMap::from([(t.clone(), TDim::Val(6))]);
    let concrete = model.clone().substitute_symbols(&subs).unwrap();
    let runnable = concrete.into_runnable().unwrap();
    let batch_out = runnable
        .run(tvec!(
            a.clone().into_dyn().into_tvalue(),
            b.clone().into_dyn().into_tvalue(),
            c.clone().into_dyn().into_tvalue()
        ))
        .unwrap();
    let batch_arr = batch_out[0].to_plain_array_view::<f32>().unwrap().to_owned();
    assert_eq!(batch_arr.shape(), &[6, 4]);

    let pulsed = PulsedModel::new(&model, t, &2.to_dim()).unwrap();
    let plan = SimplePlan::new(pulsed.into_typed().unwrap()).unwrap();
    let mut state = SimpleState::new(&plan).unwrap();

    let mut got_rows: Vec<f32> = Vec::new();
    for chunk in 0..3 {
        let a_slice = a.slice_axis(Axis(0), (chunk * 2..(chunk + 1) * 2).into()).to_owned();
        let b_slice = b.slice_axis(Axis(0), (chunk * 2..(chunk + 1) * 2).into()).to_owned();
        let c_slice = c.slice_axis(Axis(0), (chunk * 2..(chunk + 1) * 2).into()).to_owned();
        let out = state
            .run(tvec!(
                a_slice.into_dyn().into_tvalue(),
                b_slice.into_dyn().into_tvalue(),
                c_slice.into_dyn().into_tvalue()
            ))
            .unwrap();
        got_rows.extend(out[0].to_plain_array_view::<f32>().unwrap().iter());
    }

    let pulsed_arr = Array2::from_shape_vec((6, 4), got_rows).unwrap().into_dyn();
    for (b_val, p_val) in batch_arr.iter().zip(pulsed_arr.iter()) {
        assert!((b_val - p_val).abs() < 1e-4, "mismatch: batch={b_val} pulsed={p_val}");
    }
}
