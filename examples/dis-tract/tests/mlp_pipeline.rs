//! In-process proof of the partition + NNEF-ship + host-tensor-wire machinery,
//! independent of HTTP. Runs the two MLP stages through serialized boundaries.

use std::io::Cursor;

use anyhow::Result;
use tract_core::prelude::tract_data::internal::Approximation;
use tract_core::prelude::*;
use tract_distributed::{codec, models, partition, runner};

fn pipeline(full: &TypedModel, backends: [&str; 2]) -> Result<Tensor> {
    let x = models::mlp_input(256);
    let stages = partition::partition(full, &["act"])?;
    assert_eq!(stages.len(), 2, "MLP must split into 2 stages");

    // Ship each stage through NNEF bytes (proves sub-model serialization).
    let s0 = codec::model_from_bytes(&codec::model_to_bytes(&stages[0].model)?)?;
    let s1 = codec::model_from_bytes(&codec::model_to_bytes(&stages[1].model)?)?;

    // Stage 0 → serialize activation over the "wire" → stage 1.
    let mid = runner::run_once(&s0, backends[0], tvec!(x.into_tvalue()))?;
    let mid: TVec<Tensor> = mid.into_iter().map(|v| v.into_tensor()).collect();
    let mut wire = Vec::new();
    codec::write_tensors(&mut wire, &mid)?;
    let mid = codec::read_tensors(&mut Cursor::new(wire.as_slice()))?;
    let mid: TVec<TValue> = mid.into_iter().map(|t| t.into_tvalue()).collect();
    let out = runner::run_once(&s1, backends[1], mid)?;
    Ok(out[0].clone().into_tensor())
}

#[test]
fn mlp_pipeline_matches_single_machine_cpu() -> Result<()> {
    let full = models::build_mlp(256, 512, 128)?;
    let x = models::mlp_input(256);
    let reference = runner::run_once(&full, "cpu", tvec!(x.into_tvalue()))?;
    let refr = reference[0].clone().into_tensor();

    let got = pipeline(&full, ["cpu", "cpu"])?;
    let cos = runner::cosine(&got, &refr)?;
    assert!(cos > 0.9999, "cpu pipeline cosine {cos} too low");
    got.close_enough(&refr, Approximation::Approximate)?;
    Ok(())
}

#[test]
fn partition_rejects_stateful_cut() -> Result<()> {
    // "mm1" is a plain matmul → allowed; a bogus name errors; both exercised
    // lightly here. (Stateful-op rejection is covered against real KV ops in M2.)
    let full = models::build_mlp(64, 128, 32)?;
    assert!(partition::partition(&full, &["does_not_exist"]).is_err());
    assert!(partition::partition(&full, &["act"]).is_ok());
    Ok(())
}
