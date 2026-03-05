use std::time::Instant;

use anyhow::*;
use clap::Args;
use float_ord::FloatOrd;
use itertools::Itertools;
use tract_rs::prelude::tract_ndarray::prelude::*;
use tract_rs::prelude::*;

#[derive(Args, Clone)]
pub struct BeamConfig {
    /// Number of active hypotheses to keep after each pruning step
    #[arg(long, default_value_t = 4)]
    pub beam_size: usize,

    /// Number of duration candidates to expand per hypothesis
    #[arg(long, default_value_t = 2)]
    pub beam_dur_k: usize,
}

struct Beam {
    score: f32,
    tokens: Vec<usize>,
    last_frame: usize,
    dec_out: Value,
    state_0: Value,
    state_1: Value,
}

pub fn transcribe_beam(
    model: &crate::TdtModel,
    wav: &[f32],
    cfg: &BeamConfig,
) -> Result<(String, crate::DecodingStats)> {
    let mut stats = crate::DecodingStats::default();

    let samples: Value = Value::from_slice(&[1, wav.len()], wav)?;
    let len: Value = arr1(&[wav.len() as i64]).try_into()?;

    let t = Instant::now();
    let [features, feat_len] =
        model.preprocessor.run([samples, len])?.try_into().unwrap();
    stats.preprocessor.record(1, t.elapsed());

    let t = Instant::now();
    let [encoded, _lens] =
        model.encoder.run([features, feat_len])?.try_into().unwrap();
    stats.encoder.record(1, t.elapsed());

    let encoded: ArrayD<f32> = encoded.view()?.into_owned();
    let batch = encoded.shape()[0];
    let max_frames = encoded.shape()[2];
    let enc_dim = encoded.shape()[1];

    let init_token = Value::from_slice(&[1, 1], &[model.blank_id as i32])?;
    let init_s0: Value = Array3::<f32>::zeros([2, 1, 640]).try_into()?;
    let init_s1: Value = Array3::<f32>::zeros([2, 1, 640]).try_into()?;
    let t = Instant::now();
    let [dec_out, state_0, state_1] =
        model.decoder.run([init_token, init_s0, init_s1])?.try_into().unwrap();
    stats.decoder.record(batch, t.elapsed());

    let mut all_beams: Vec<Beam> = vec![Beam {
        score: 0.0,
        tokens: vec![],
        last_frame: 0,
        dec_out,
        state_0,
        state_1,
    }];

    for frame_ix in 0..max_frames {
        let mut hyps: Vec<Beam> = Vec::new();
        let mut kept: Vec<Beam> = Vec::new();
        for b in all_beams.drain(..) {
            if b.last_frame == frame_ix {
                hyps.push(b);
            } else {
                kept.push(b);
            }
        }

        while !hyps.is_empty() {
            let b = hyps.len();

            // 1. JOINT: single call batched over all B active hypotheses
            let frame_batch: Value = {
                let enc_arr = encoded.view();
                Array3::<f32>::from_shape_fn((b, enc_dim, 1), |(_, e, _)| {
                    enc_arr[[0, e, frame_ix]]
                })
                .try_into()?
            };
            let dec_out_batch: Value = {
                let views: Vec<_> = hyps
                    .iter()
                    .map(|h| h.dec_out.view::<f32>())
                    .collect::<Result<Vec<_>>>()?;
                let hidden = views[0].shape()[1]; // dec_out is [1, hidden, 1]
                Array3::<f32>::from_shape_fn((b, hidden, 1), |(bi, h, _)| views[bi][[0, h, 0]])
                    .try_into()?
            };
            let t = Instant::now();
            let [logits_b] =
                model.joint.run([frame_batch, dec_out_batch])?.try_into().unwrap();
            stats.joint.record(b, t.elapsed());

            // 2. Per-hyp: token scores + duration expansions into kept
            let mut per_hyp_token_scores: Vec<Vec<(usize, f32)>> = Vec::with_capacity(b);
            let mut per_hyp_best_dur: Vec<(usize, f32)> = Vec::with_capacity(b);
            {
                let logits_arr = logits_b.view::<f32>()?; // [b, vocab+dur]
                for bi in 0..b {
                    let row = logits_arr.index_axis(Axis(0), bi);
                    let row_slice = row.as_slice().unwrap();
                    let log_probs = crate::log_softmax(&row_slice[0..=model.blank_id]);
                    let dur_log_probs = crate::log_softmax(&row_slice[model.blank_id + 1..]);

                    let mut ts: Vec<(usize, f32)> =
                        (0..model.blank_id).map(|ti| (ti, log_probs[ti])).collect();
                    ts.sort_by(|a, b| FloatOrd(b.1).cmp(&FloatOrd(a.1)));
                    ts.truncate(cfg.beam_size);
                    per_hyp_token_scores.push(ts);

                    let best_dur = dur_log_probs.iter().enumerate()
                        .max_by_key(|&(_, &v)| FloatOrd(v)).map(|(i, _)| i).unwrap_or(0);
                    per_hyp_best_dur.push((best_dur, dur_log_probs[best_dur]));

                    let mut ds: Vec<(usize, f32)> =
                        (1..dur_log_probs.len()).map(|di| (di, dur_log_probs[di])).collect();
                    ds.sort_by(|a, b| FloatOrd(b.1).cmp(&FloatOrd(a.1)));
                    ds.truncate(cfg.beam_dur_k);
                    for (di, dlp) in ds {
                        kept.push(Beam {
                            score: hyps[bi].score + log_probs[model.blank_id] + dlp,
                            tokens: hyps[bi].tokens.clone(),
                            last_frame: frame_ix + di,
                            dec_out: hyps[bi].dec_out.clone(),
                            state_0: hyps[bi].state_0.clone(),
                            state_1: hyps[bi].state_1.clone(),
                        });
                    }
                }
            } // logits_arr dropped

            // 3. DECODER: single call batched over all N token expansions
            let expansion_hyp_idxs: Vec<usize> = per_hyp_token_scores
                .iter()
                .enumerate()
                .flat_map(|(bi, ts)| std::iter::repeat(bi).take(ts.len()))
                .collect();
            let token_ids: Vec<i32> = per_hyp_token_scores
                .iter()
                .flat_map(|ts| ts.iter().map(|&(ti, _)| ti as i32))
                .collect();
            let n = token_ids.len();

            let tokens_batch: Value =
                Array2::<i32>::from_shape_fn((n, 1), |(i, _)| token_ids[i]).try_into()?;
            let s0_batch: Value = {
                let views: Vec<_> = hyps
                    .iter()
                    .map(|h| h.state_0.view::<f32>())
                    .collect::<Result<Vec<_>>>()?;
                Array3::<f32>::from_shape_fn((2, n, 640), |(l, i, h)| {
                    views[expansion_hyp_idxs[i]][[l, 0, h]]
                })
                .try_into()?
            };
            let s1_batch: Value = {
                let views: Vec<_> = hyps
                    .iter()
                    .map(|h| h.state_1.view::<f32>())
                    .collect::<Result<Vec<_>>>()?;
                Array3::<f32>::from_shape_fn((2, n, 640), |(l, i, h)| {
                    views[expansion_hyp_idxs[i]][[l, 0, h]]
                })
                .try_into()?
            };

            let t = Instant::now();
            let [dec_out_b, s0_b, s1_b] =
                model.decoder.run([tokens_batch, s0_batch, s1_batch])?.try_into().unwrap();
            stats.decoder.record(n, t.elapsed());

            // 4. Slice and build new beams
            let new_hyps: Vec<Beam> = {
                let dec_arr = dec_out_b.view::<f32>()?; // [N, hidden]
                let s0_arr = s0_b.view::<f32>()?; // [2, N, 640]
                let s1_arr = s1_b.view::<f32>()?; // [2, N, 640]
                let mut out = Vec::with_capacity(n);
                let mut i = 0;
                for (bi, ts) in per_hyp_token_scores.iter().enumerate() {
                    let (best_dur, best_dur_lp) = per_hyp_best_dur[bi];
                    for &(ti, lp) in ts {
                        let new_dec_out: Value =
                            dec_arr.slice_axis(Axis(0), (i..i + 1).into()).try_into()?;
                        let new_s0: Value =
                            s0_arr.slice_axis(Axis(1), (i..i + 1).into()).try_into()?;
                        let new_s1: Value =
                            s1_arr.slice_axis(Axis(1), (i..i + 1).into()).try_into()?;
                        let mut new_tokens = hyps[bi].tokens.clone();
                        new_tokens.push(ti);
                        out.push(Beam {
                            score: hyps[bi].score + lp + best_dur_lp,
                            tokens: new_tokens,
                            last_frame: frame_ix + best_dur,
                            dec_out: new_dec_out,
                            state_0: new_s0,
                            state_1: new_s1,
                        });
                        i += 1;
                    }
                }
                out
            };
            hyps = new_hyps;

            // 5. Fuse duplicates then prune to BEAM_SIZE.
            // After sorting by score, the first occurrence of each (tokens, last_frame)
            // key is the best one; retain only that first occurrence.
            let mut all: Vec<Beam> = hyps.drain(..).chain(kept.drain(..)).collect();
            all.sort_by(|a, b| FloatOrd(b.score).cmp(&FloatOrd(a.score)));
            let mut seen = std::collections::HashSet::<(Vec<usize>, usize)>::new();
            all.retain(|h| seen.insert((h.tokens.clone(), h.last_frame)));
            all.truncate(cfg.beam_size);
            for b in all {
                if b.last_frame == frame_ix {
                    hyps.push(b);
                } else {
                    kept.push(b);
                }
            }
        }

        all_beams = kept;
    }

    let best = all_beams
        .into_iter()
        .max_by_key(|b| FloatOrd(b.score))
        .ok_or_else(|| anyhow!("no beams survived"))?;
    Ok((best.tokens.into_iter().map(|t| model.vocab[t].as_str()).join(""), stats))
}
