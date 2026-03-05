use std::time::Instant;

use anyhow::*;
use clap::Args;
use float_ord::FloatOrd;
use itertools::Itertools;
use tract_rs::prelude::tract_ndarray::prelude::*;
use tract_rs::prelude::*;

#[derive(Args, Clone)]
pub struct AlsdConfig {
    /// Beam width for ALSD decoding
    #[arg(long, default_value_t = 4)]
    pub alsd_beam_size: usize,

    /// Duration candidates per hypothesis in ALSD
    #[arg(long, default_value_t = 2)]
    pub alsd_beam_dur_k: usize,

    /// Maximum expected output token count (U_max in the paper)
    #[arg(long, default_value_t = 50)]
    pub alsd_u_max: usize,
}

struct AlsdHyp {
    score: f32,
    tokens: Vec<usize>,
    current_frame: usize,
    dec_out: Value,
    state_0: Value,
    state_1: Value,
}

pub fn transcribe_alsd(
    model: &crate::TdtModel,
    wav: &[f32],
    cfg: &AlsdConfig,
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
    let batch = encoded.shape()[0];
    let max_frames = encoded.shape()[2];
    let enc_dim = encoded.shape()[1];

    let init_token = Value::from_slice(&[1, 1], &[model.blank_id as i32])?;
    let init_s0: Value = Array3::<f32>::zeros([2, 1, 640]).try_into()?;
    let init_s1: Value = Array3::<f32>::zeros([2, 1, 640]).try_into()?;
    let t = Instant::now();
    let [dec_out, state_0, state_1] = model.run_decoder(init_token, init_s0, init_s1)?;
    stats.decoder.record(batch, t.elapsed());

    let mut beam: Vec<AlsdHyp> = vec![AlsdHyp {
        score: 0.0,
        tokens: vec![],
        current_frame: 0,
        dec_out,
        state_0,
        state_1,
    }];

    let mut final_hyps: Vec<AlsdHyp> = Vec::new();

    // Outer loop: alignment steps 0 .. T + U_max.
    // All hypotheses in `beam` are at the same alignment step.
    // At each step every active hyp (current_frame < T) gets one expansion
    // (blank+duration OR one non-blank token), so a single step advances
    // the alignment length by exactly 1.
    for _step in 0..(max_frames + cfg.alsd_u_max) {
        // Separate completed hypotheses (exhausted all frames) from active ones.
        let mut active: Vec<AlsdHyp> = Vec::new();
        for h in beam.drain(..) {
            if h.current_frame >= max_frames {
                final_hyps.push(h);
            } else {
                active.push(h);
            }
        }
        if active.is_empty() {
            break;
        }

        let b = active.len();

        // 1. JOINT: batched over all active hypotheses (each at its own frame).
        let frame_batch: Value = {
            let enc_arr = encoded.view();
            Array3::<f32>::from_shape_fn((b, enc_dim, 1), |(bi, e, _)| {
                enc_arr[[0, e, active[bi].current_frame]]
            })
            .try_into()?
        };
        let dec_out_batch: Value = {
            let views: Vec<_> = active
                .iter()
                .map(|h| h.dec_out.view::<f32>())
                .collect::<Result<Vec<_>>>()?;
            let hidden = views[0].shape()[1];
            Array3::<f32>::from_shape_fn((b, hidden, 1), |(bi, h, _)| views[bi][[0, h, 0]])
                .try_into()?
        };
        let t = Instant::now();
        let logits_b = model.run_joint(frame_batch, dec_out_batch)?;
        stats.joint.record(b, t.elapsed());

        // 2. Per-hyp: blank+duration children (no decoder update) and
        //    non-blank token candidates (need decoder update).
        let mut next: Vec<AlsdHyp> = Vec::new();
        let mut per_hyp_token_scores: Vec<Vec<(usize, f32)>> = Vec::with_capacity(b);
        let mut per_hyp_best_dur: Vec<(usize, f32)> = Vec::with_capacity(b);
        {
            let logits_arr = logits_b.view::<f32>()?;
            for bi in 0..b {
                let row = logits_arr.index_axis(Axis(0), bi);
                let row_slice = row.as_slice().unwrap();
                let log_probs = crate::log_softmax(&row_slice[0..=model.blank_id]);
                let dur_log_probs = crate::log_softmax(&row_slice[model.blank_id + 1..]);

                let best_dur = dur_log_probs.iter().enumerate()
                    .max_by_key(|&(_, &v)| FloatOrd(v)).map(|(i, _)| i).unwrap_or(0);
                per_hyp_best_dur.push((best_dur, dur_log_probs[best_dur]));

                // Blank + top-k duration expansions (decoder state unchanged).
                let mut ds: Vec<(usize, f32)> =
                    (1..dur_log_probs.len()).map(|di| (di, dur_log_probs[di])).collect();
                ds.sort_by(|a, b| FloatOrd(b.1).cmp(&FloatOrd(a.1)));
                ds.truncate(cfg.alsd_beam_dur_k);
                for (di, dlp) in ds {
                    next.push(AlsdHyp {
                        score: active[bi].score + log_probs[model.blank_id] + dlp,
                        tokens: active[bi].tokens.clone(),
                        current_frame: active[bi].current_frame + di,
                        dec_out: active[bi].dec_out.clone(),
                        state_0: active[bi].state_0.clone(),
                        state_1: active[bi].state_1.clone(),
                    });
                }

                // Top-k non-blank token candidates (decoder step needed).
                let mut ts: Vec<(usize, f32)> =
                    (0..model.blank_id).map(|ti| (ti, log_probs[ti])).collect();
                ts.sort_by(|a, b| FloatOrd(b.1).cmp(&FloatOrd(a.1)));
                ts.truncate(cfg.alsd_beam_size);
                per_hyp_token_scores.push(ts);
            }
        }

        // 3. DECODER: single batched call over all non-blank expansions.
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

        if n > 0 {
            let tokens_batch: Value =
                Array2::<i32>::from_shape_fn((n, 1), |(i, _)| token_ids[i]).try_into()?;
            let s0_batch: Value = {
                let views: Vec<_> = active
                    .iter()
                    .map(|h| h.state_0.view::<f32>())
                    .collect::<Result<Vec<_>>>()?;
                Array3::<f32>::from_shape_fn((2, n, 640), |(l, i, h)| {
                    views[expansion_hyp_idxs[i]][[l, 0, h]]
                })
                .try_into()?
            };
            let s1_batch: Value = {
                let views: Vec<_> = active
                    .iter()
                    .map(|h| h.state_1.view::<f32>())
                    .collect::<Result<Vec<_>>>()?;
                Array3::<f32>::from_shape_fn((2, n, 640), |(l, i, h)| {
                    views[expansion_hyp_idxs[i]][[l, 0, h]]
                })
                .try_into()?
            };

            let t = Instant::now();
            let [dec_out_b, s0_b, s1_b] = model.run_decoder(tokens_batch, s0_batch, s1_batch)?;
            stats.decoder.record(n, t.elapsed());

            let dec_arr = dec_out_b.view::<f32>()?;
            let s0_arr = s0_b.view::<f32>()?;
            let s1_arr = s1_b.view::<f32>()?;
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
                    let mut new_tokens = active[bi].tokens.clone();
                    new_tokens.push(ti);
                    next.push(AlsdHyp {
                        score: active[bi].score + lp + best_dur_lp,
                        tokens: new_tokens,
                        current_frame: active[bi].current_frame + best_dur,
                        dec_out: new_dec_out,
                        state_0: new_s0,
                        state_1: new_s1,
                    });
                    i += 1;
                }
            }
        }

        // 4. Prune: sort by score, deduplicate on (tokens, frame), keep top beam_size.
        // Two hyps with identical tokens and frame have identical decoder state;
        // keep only the highest-scoring one.
        next.sort_by(|a, b| FloatOrd(b.score).cmp(&FloatOrd(a.score)));
        let mut seen = std::collections::HashSet::<(Vec<usize>, usize)>::new();
        next.retain(|h| seen.insert((h.tokens.clone(), h.current_frame)));
        next.truncate(cfg.alsd_beam_size);
        beam = next;
    }

    // Any hyps remaining in beam that didn't finish (step limit hit) go to final.
    final_hyps.extend(beam);

    let best = final_hyps
        .into_iter()
        .max_by_key(|h| FloatOrd(h.score))
        .ok_or_else(|| anyhow!("no hypotheses survived"))?;
    Ok((best.tokens.into_iter().map(|t| model.vocab[t].as_str()).join(""), stats))
}
