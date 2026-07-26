//! Per-sequence KV in a stage: sequences in flight together must not see each
//! other's cache, and a freed sequence must leave nothing behind.

use anyhow::Result;
use tract_core::dims;
use tract_core::prelude::*;
use tract_distributed::llm::StageState;
use tract_distributed::shard_graph::shard_io_roles;

const D: usize = 4;

/// A stand-in for a decoder block's KV contract: append the step's activation to
/// the resident cache and report the cache's running sum. With an all-ones
/// activation the sum is the cache depth, which is what makes "whose KV did this
/// step use?" observable at all.
fn cache_model() -> Result<TypedModel> {
    let mut m = TypedModel::default();
    let p = m.sym("P");
    let x = m.add_source("x", f32::fact(dims!(1, 1, D)))?;
    let c = m.add_source("in_cache_key_0", f32::fact(dims!(1, p, D)))?;
    let cat =
        m.wire_node("out_cache_key_0", tract_core::ops::array::TypedConcat { axis: 1 }, &[c, x])?
            [0];
    let sum = m.wire_node(
        "depth",
        tract_core::ops::nn::Reduce { axes: tvec![1], reducer: tract_core::ops::nn::Reducer::Sum },
        &[cat],
    )?[0];
    m.select_output_outlets(&[sum, cat])?;
    m.into_decluttered()
}

fn stage() -> Result<StageState> {
    let model = cache_model()?;
    let (inputs, outputs) = shard_io_roles(&model)?;
    StageState::new(model, "cpu", inputs, outputs)
}

/// One step for `seq`, returning the cache depth the stage ran against.
fn depth(st: &mut StageState, seq: u64) -> Result<usize> {
    let x = Tensor::from_shape(&[1, 1, D], &[1.0f32; D])?;
    let out = st.step(seq, tvec!(x))?;
    let sums = out[0].cast_to::<f32>()?;
    let view = sums.view();
    Ok(view.as_slice::<f32>()?[0] as usize)
}

#[test]
fn sequences_do_not_share_a_cache() -> Result<()> {
    let mut st = stage()?;
    assert_eq!(depth(&mut st, 1)?, 1);
    assert_eq!(depth(&mut st, 1)?, 2);
    // A sequence seen for the first time starts empty, whatever else is resident.
    assert_eq!(depth(&mut st, 2)?, 1);
    assert_eq!(depth(&mut st, 2)?, 2);
    // Interleaving does not disturb the first sequence's context.
    assert_eq!(depth(&mut st, 1)?, 3);
    assert_eq!(st.resident_seqs(), 2);
    Ok(())
}

#[test]
fn reset_clears_one_sequence_only() -> Result<()> {
    let mut st = stage()?;
    depth(&mut st, 1)?;
    depth(&mut st, 1)?;
    depth(&mut st, 2)?;
    st.reset(1)?;
    assert_eq!(depth(&mut st, 1)?, 1);
    assert_eq!(depth(&mut st, 2)?, 2);
    Ok(())
}

#[test]
fn freeing_a_sequence_releases_its_cache() -> Result<()> {
    let mut st = stage()?;
    depth(&mut st, 1)?;
    depth(&mut st, 1)?;
    depth(&mut st, 2)?;
    assert_eq!(st.resident_seqs(), 2);

    st.free(1);
    assert_eq!(st.resident_seqs(), 1);
    // The id is reusable: it comes back as a fresh sequence, not a resumed one.
    assert_eq!(depth(&mut st, 1)?, 1);
    assert_eq!(depth(&mut st, 2)?, 2);

    st.free(404);
    assert_eq!(st.resident_seqs(), 2);
    Ok(())
}
