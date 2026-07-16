//! M2 in-process validation on a real decoder LLM (run with DISTRACT_MODEL).
//! Proves the role-aware split + resident-KV loop against a single-machine
//! reference, for both prefill parity and multi-step greedy decode.
//!
//! DISTRACT_MODEL=/path/openelm.nnef.tgz cargo test -p tract-distributed \
//!   --test llm_pipeline -- --ignored --nocapture

use anyhow::Result;
use tract_core::prelude::*;
use tract_distributed::llm::{StageState, full_io_roles, load_model, partition_stages};
use tract_distributed::runner;

const SPLIT_LAYER: usize = 8;

/// The model plus its regular (non-cache) output count, loaded through the same
/// path the binaries use so node naming and KV I/O match.
fn load_full() -> Result<(TypedModel, usize)> {
    let path = std::env::var("DISTRACT_MODEL").expect("set DISTRACT_MODEL");
    load_model(&path)
}

fn ids(tokens: &[i64]) -> Tensor {
    Tensor::from_shape(&[1, tokens.len()], tokens).unwrap()
}

fn argmax(logits: &Tensor) -> usize {
    let v = logits.cast_to::<f32>().unwrap();
    let v = v.view();
    let s = v.as_slice::<f32>().unwrap();
    s.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
}

#[test]
#[ignore]
fn llm_prefill_parity() -> Result<()> {
    let (full, n_regular) = load_full()?;
    let (fin, fout) = full_io_roles(&full, n_regular)?;
    let mut reference = StageState::new(full.clone(), "cpu", fin, fout)?;
    let prompt = ids(&[9, 3, 128, 42]);
    let ref_logits = reference.step(tvec!(prompt.clone()))?[0].clone();

    let stages = partition_stages(&full, &[SPLIT_LAYER], n_regular)?;
    assert_eq!(stages.len(), 2);
    let [s0, s1]: [_; 2] = stages.try_into().ok().unwrap();
    let mut stage0 = StageState::new(s0.model, "cpu", s0.inputs, s0.outputs)?;
    let mut stage1 = StageState::new(s1.model, "cpu", s1.inputs, s1.outputs)?;

    let residual = stage0.step(tvec!(prompt))?;
    let got_logits = stage1.step(residual)?[0].clone();

    let gf = got_logits.cast_to::<f32>()?;
    let rf = ref_logits.cast_to::<f32>()?;
    let cos = runner::cosine(gf.as_ref(), rf.as_ref())?;
    println!("prefill logits cosine = {cos:.6}");
    println!("argmax  ref={}  dist={}", argmax(&ref_logits), argmax(&got_logits));
    assert_eq!(argmax(&got_logits), argmax(&ref_logits), "next-token mismatch");
    assert!(cos > 0.999, "prefill cosine {cos} too low");
    Ok(())
}

#[test]
#[ignore]
fn llm_greedy_decode_parity() -> Result<()> {
    let (full, n_regular) = load_full()?;
    let (fin, fout) = full_io_roles(&full, n_regular)?;
    let mut reference = StageState::new(full.clone(), "cpu", fin, fout)?;

    let stages = partition_stages(&full, &[SPLIT_LAYER], n_regular)?;
    let [s0, s1]: [_; 2] = stages.try_into().ok().unwrap();
    let mut stage0 = StageState::new(s0.model, "cpu", s0.inputs, s0.outputs)?;
    let mut stage1 = StageState::new(s1.model, "cpu", s1.inputs, s1.outputs)?;

    // Prompt, then 8 greedy decode steps; assert token sequences agree.
    let prompt = vec![9i64, 3, 128, 42];
    let mut ref_next = argmax(&reference.step(tvec!(ids(&prompt)))?[0].clone()) as i64;
    let mut dist_next = argmax(&stage1.step(stage0.step(tvec!(ids(&prompt)))?)?[0].clone()) as i64;
    assert_eq!(ref_next, dist_next, "prompt-step token mismatch");

    for step in 0..8 {
        ref_next = argmax(&reference.step(tvec!(ids(&[ref_next])))?[0].clone()) as i64;
        let d = stage1.step(stage0.step(tvec!(ids(&[dist_next])))?)?[0].clone();
        dist_next = argmax(&d) as i64;
        println!("decode {step}: ref={ref_next} dist={dist_next}");
        assert_eq!(ref_next, dist_next, "decode token mismatch at step {step}");
    }
    Ok(())
}
