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
    pub alsd_dur_beam_k: usize,

    /// Max non-blank tokens emitted per frame per hypothesis
    #[arg(long, default_value_t = 10)]
    pub alsd_max_symbols_per_frame: usize,
}

struct AlsdHyp {
    score: f32,
    tokens: Vec<usize>,
    current_frame: usize,
    symbols_this_frame: usize,
    dec_out: Value,
    state_0: Value,
    state_1: Value,
}

pub fn transcribe_alsd(
    model: &crate::TdtModel,
    wav: &[f32],
    cfg: &AlsdConfig,
) -> Result<(String, crate::CallStats, crate::CallStats)> {
    let samples: Value = Value::from_slice(&[1, wav.len()], wav)?;
    let len: Value = arr1(&[wav.len() as i64]).try_into()?;

    let [features, feat_len] =
        model.preprocessor.run([samples, len])?.try_into().unwrap();
    let [encoded, _lens] =
        model.encoder.run([features, feat_len])?.try_into().unwrap();

    let encoded: ArrayD<f32> = encoded.view()?.into_owned();
    let batch = encoded.shape()[0];
    let max_frames = encoded.shape()[2];
    let enc_dim = encoded.shape()[1];

    let mut decoder_stats = crate::CallStats::default();
    let mut joint_stats = crate::CallStats::default();

    let init_token = Value::from_slice(&[1, 1], &[0i32])?;
    let init_s0: Value = Array3::<f32>::zeros([2, 1, 640]).try_into()?;
    let init_s1: Value = Array3::<f32>::zeros([2, 1, 640]).try_into()?;
    let t = Instant::now();
    let [dec_out, state_0, state_1] =
        model.decoder.run([init_token, init_s0, init_s1])?.try_into().unwrap();
    decoder_stats.record(batch, t.elapsed());

    let mut beam: Vec<AlsdHyp> = vec![AlsdHyp {
        score: 0.0,
        tokens: vec![],
        current_frame: 0,
        symbols_this_frame: 0,
        dec_out,
        state_0,
        state_1,
    }];

    loop {
        // Split into active (still have frames to consume) and completed
        let mut active: Vec<AlsdHyp> = Vec::new();
        let mut completed: Vec<AlsdHyp> = Vec::new();
        for h in beam.drain(..) {
            if h.current_frame < max_frames {
                active.push(h);
            } else {
                completed.push(h);
            }
        }

        if active.is_empty() {
            beam = completed;
            break;
        }

        let b = active.len();

        // 1. JOINT: each hypothesis uses its own current_frame
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
            let hidden = views[0].shape()[1]; // dec_out is [1, hidden, 1]
            Array3::<f32>::from_shape_fn((b, hidden, 1), |(bi, h, _)| views[bi][[0, h, 0]])
                .try_into()?
        };
        let t = Instant::now();
        let [logits_b] =
            model.joint.run([frame_batch, dec_out_batch])?.try_into().unwrap();
        joint_stats.record(b, t.elapsed());

        // 2. Per-hyp: blank+duration → next; non-blank → expansion list
        let mut next: Vec<AlsdHyp> = completed;
        let mut per_hyp_token_scores: Vec<Vec<(usize, f32)>> = Vec::with_capacity(b);
        {
            let logits_arr = logits_b.view::<f32>()?; // [b, vocab+dur]
            for bi in 0..b {
                let row = logits_arr.index_axis(Axis(0), bi);
                let row_slice = row.as_slice().unwrap();
                let log_probs = crate::log_softmax(&row_slice[0..=model.blank_id]);
                let dur_log_probs = crate::log_softmax(&row_slice[model.blank_id + 1..]);

                // Blank + duration expansions advance the frame and reset symbol count
                let mut ds: Vec<(usize, f32)> =
                    (1..dur_log_probs.len()).map(|di| (di, dur_log_probs[di])).collect();
                ds.sort_by(|a, b| FloatOrd(b.1).cmp(&FloatOrd(a.1)));
                ds.truncate(cfg.alsd_dur_beam_k);
                for (di, dlp) in ds {
                    next.push(AlsdHyp {
                        score: active[bi].score + log_probs[model.blank_id] + dlp,
                        tokens: active[bi].tokens.clone(),
                        current_frame: active[bi].current_frame + di,
                        symbols_this_frame: 0,
                        dec_out: active[bi].dec_out.clone(),
                        state_0: active[bi].state_0.clone(),
                        state_1: active[bi].state_1.clone(),
                    });
                }

                // Non-blank expansions stay on the same frame (symbol count checked)
                if active[bi].symbols_this_frame < cfg.alsd_max_symbols_per_frame {
                    let mut ts: Vec<(usize, f32)> =
                        (0..model.blank_id).map(|ti| (ti, log_probs[ti])).collect();
                    ts.sort_by(|a, b| FloatOrd(b.1).cmp(&FloatOrd(a.1)));
                    ts.truncate(cfg.alsd_beam_size);
                    per_hyp_token_scores.push(ts);
                } else {
                    per_hyp_token_scores.push(vec![]);
                }
            }
        } // logits_arr dropped

        // 3. DECODER: single batched call over all non-blank expansions
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
            let [dec_out_b, s0_b, s1_b] =
                model.decoder.run([tokens_batch, s0_batch, s1_batch])?.try_into().unwrap();
            decoder_stats.record(n, t.elapsed());

            // 4. Slice per-expansion outputs and push into next
            let dec_arr = dec_out_b.view::<f32>()?; // [N, hidden, 1]
            let s0_arr = s0_b.view::<f32>()?;       // [2, N, 640]
            let s1_arr = s1_b.view::<f32>()?;       // [2, N, 640]
            let mut i = 0;
            for (bi, ts) in per_hyp_token_scores.iter().enumerate() {
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
                        score: active[bi].score + lp,
                        tokens: new_tokens,
                        current_frame: active[bi].current_frame,
                        symbols_this_frame: active[bi].symbols_this_frame + 1,
                        dec_out: new_dec_out,
                        state_0: new_s0,
                        state_1: new_s1,
                    });
                    i += 1;
                }
            }
        }

        // 5. Global prune
        next.sort_by(|a, b| FloatOrd(b.score).cmp(&FloatOrd(a.score)));
        next.truncate(cfg.alsd_beam_size);
        beam = next;
    }

    let best = beam
        .into_iter()
        .max_by_key(|b| FloatOrd(b.score))
        .ok_or_else(|| anyhow!("no beams survived"))?;
    Ok((best.tokens.into_iter().map(|t| model.vocab[t].as_str()).join(""), decoder_stats, joint_stats))
}
