use std::time::Instant;

use anyhow::*;
use itertools::Itertools;
use tract_rs::prelude::tract_ndarray::prelude::*;
use tract_rs::prelude::*;

pub fn transcribe_greedy(
    model: &crate::TdtModel,
    wav: &[f32],
) -> Result<(String, crate::DecodingStats)> {
    let mut stats = crate::DecodingStats::default();

    let samples: Value = Value::from_slice(&[1, wav.len()], wav)?;
    let len: Value = arr1(&[wav.len() as i64]).try_into()?;

    let t = Instant::now();
    let [features, feat_len] = model.run_preprocessor(samples, len)?;
    stats.preprocessor.record(1, t.elapsed());

    let t = Instant::now();
    let [encoded, _lens] = model.run_encoder(features, feat_len)?;
    stats.encoder.record(1, t.elapsed());

    let encoded: ArrayD<f32> = encoded.view()?.into_owned();
    let max_frames = encoded.shape()[2];
    let max_len = max_frames * 6 + 10;

    let mut hyp = vec![];
    let mut frame_ix = 0;
    let mut token = Value::from_slice(&[1, 1], &[model.blank_id as i32])?;
    let mut state_0: Value = Array3::<f32>::zeros([2, 1, 640]).try_into()?;
    let mut state_1: Value = Array3::<f32>::zeros([2, 1, 640]).try_into()?;

    let t = Instant::now();
    [token, state_0, state_1] = model.run_decoder(token, state_0, state_1)?;
    stats.decoder.record(1, t.elapsed());

    while hyp.len() < max_len && frame_ix < max_frames {
        let frame: Value =
            encoded.slice_axis(Axis(2), (frame_ix..frame_ix + 1).into()).try_into()?;
        let t = Instant::now();
        let logits = model.run_joint(frame, token.clone())?;
        stats.joint.record(1, t.elapsed());
        let logits = logits.view::<f32>()?;
        let logits = logits.as_slice().unwrap();
        let token_id = crate::argmax(&logits[0..model.blank_id + 1]).unwrap();
        let dur = crate::argmax(&logits[model.blank_id + 1..]).unwrap_or(0);
        if token_id == model.blank_id {
            frame_ix += dur.max(1);
        } else {
            hyp.push(token_id);
            frame_ix += dur;
            token = Value::from_slice(&[1, 1], &[token_id as i32])?;
            let t = Instant::now();
            [token, state_0, state_1] = model.run_decoder(token, state_0, state_1)?;
            stats.decoder.record(1, t.elapsed());
        }
    }

    Ok((hyp.into_iter().map(|t| model.vocab[t].as_str()).join(""), stats))
}
