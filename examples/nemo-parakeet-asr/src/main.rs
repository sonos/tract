use std::fs::File;
use std::path::Path;
use std::time::Instant;

use anyhow::*;
use float_ord::FloatOrd;
use itertools::Itertools;
use tract_rs::prelude::tract_ndarray::prelude::*;
use tract_rs::prelude::*;
use tract_rs::Nnef;

#[derive(Default)]
struct CallStats {
    calls: u32,
    total_batch: u64,
    total_us: u64,
}

impl CallStats {
    fn record(&mut self, batch: usize, elapsed: std::time::Duration) {
        self.calls += 1;
        self.total_batch += batch as u64;
        self.total_us += elapsed.as_micros() as u64;
    }
}

impl std::fmt::Debug for CallStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let avg_batch =
            if self.calls == 0 { 0.0 } else { self.total_batch as f64 / self.calls as f64 };
        let avg_ms =
            if self.calls == 0 { 0.0 } else { self.total_us as f64 / self.calls as f64 / 1000.0 };
        let total_ms = self.total_us as f64 / 1000.0;
        write!(
            f,
            "calls={:5}  avg_batch={avg_batch:.1}  avg={avg_ms:.3}ms  total={total_ms:.1}ms",
            self.calls
        )
    }
}

fn argmax(slice: &[f32]) -> Option<usize> {
    slice.into_iter().position_max_by_key(|x| FloatOrd(**x))
}

fn log_softmax(xs: &[f32]) -> Vec<f32> {
    let max = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let lse = xs.iter().map(|&x| (x - max).exp()).sum::<f32>().ln();
    xs.iter().map(|&x| x - max - lse).collect()
}

struct TdtModel {
    preprocessor: Runnable,
    encoder: Runnable,
    decoder: Runnable,
    joint: Runnable,
    vocab: Vec<String>,
    blank_id: usize,
}

impl TdtModel {
    fn load(model_dir: impl AsRef<Path>, nnef: &Nnef, gpu: &Runtime) -> Result<TdtModel> {
        let model_dir = model_dir.as_ref();
        let config: serde_json::Value =
            serde_json::from_reader(File::open(model_dir.join("model_config.json"))?)?;
        let blank_id =
            config.pointer("/decoder/vocab_size").unwrap().as_i64().unwrap() as usize;
        let vocab = config.pointer("/joint/vocabulary").unwrap().as_array().unwrap();
        let vocab: Vec<String> = vocab.iter().map(|v| v.as_str().unwrap().to_owned()).collect();

        let preprocessor =
            nnef.load(model_dir.join("preprocessor.nnef.tgz"))?.into_runnable()?;

        let mut encoder = nnef.load(model_dir.join("encoder.nnef.tgz"))?;
        encoder.transform("transformers-detect-all")?;
        let encoder = gpu.prepare(encoder)?;

        let decoder = nnef.load(model_dir.join("decoder.nnef.tgz"))?;
        let decoder = gpu.prepare(decoder)?;

        let joint = nnef.load(model_dir.join("joint.nnef.tgz"))?;
        let joint = gpu.prepare(joint)?;

        Ok(TdtModel { preprocessor, encoder, decoder, joint, vocab, blank_id })
    }

    fn transcribe_greedy(&self, wav: &[f32]) -> Result<(String, CallStats, CallStats)> {
        let samples: Value = Value::from_slice(&[1, wav.len()], wav)?;
        let len: Value = arr1(&[wav.len() as i64]).try_into()?;

        let [features, feat_len] =
            self.preprocessor.run([samples, len])?.try_into().unwrap();
        let [encoded, _lens] =
            self.encoder.run([features, feat_len])?.try_into().unwrap();

        let encoded: ArrayD<f32> = encoded.view()?.into_owned();
        let batch = encoded.shape()[0];

        let max_frames = encoded.shape()[2];
        let max_len = max_frames * 6 + 10;

        let mut decoder_stats = CallStats::default();
        let mut joint_stats = CallStats::default();

        let mut hyp = vec![];
        let mut frame_ix = 0;
        let mut token = Value::from_slice(&[1, 1], &[0i32])?;
        let mut state_0: Value = Array3::<f32>::zeros([2, 1, 640]).try_into()?;
        let mut state_1: Value = Array3::<f32>::zeros([2, 1, 640]).try_into()?;

        let t = Instant::now();
        [token, state_0, state_1] =
            self.decoder.run([token, state_0, state_1])?.try_into().unwrap();
        decoder_stats.record(batch, t.elapsed());

        while hyp.len() < max_len && frame_ix < max_frames {
            let frame: Value =
                encoded.slice_axis(Axis(2), (frame_ix..frame_ix + 1).into()).try_into()?;
            let t = Instant::now();
            let [logits] = self.joint.run([frame, token.clone()])?.try_into().unwrap();
            joint_stats.record(batch, t.elapsed());
            let logits = logits.view::<f32>()?;
            let logits = logits.as_slice().unwrap();
            let token_id = argmax(&logits[0..self.blank_id + 1]).unwrap();
            if token_id == self.blank_id {
                frame_ix += argmax(&logits[self.blank_id + 1..]).unwrap_or(0).max(1);
            } else {
                hyp.push(token_id);
                token = Value::from_slice(&[1, 1], &[token_id as i32])?;
                let t = Instant::now();
                [token, state_0, state_1] =
                    self.decoder.run([token, state_0, state_1])?.try_into().unwrap();
                decoder_stats.record(batch, t.elapsed());
            }
        }

        Ok((hyp.into_iter().map(|t| self.vocab[t].as_str()).join(""), decoder_stats, joint_stats))
    }

    fn transcribe_beam(&self, wav: &[f32]) -> Result<(String, CallStats, CallStats)> {
        let samples: Value = Value::from_slice(&[1, wav.len()], wav)?;
        let len: Value = arr1(&[wav.len() as i64]).try_into()?;

        let [features, feat_len] =
            self.preprocessor.run([samples, len])?.try_into().unwrap();
        let [encoded, _lens] =
            self.encoder.run([features, feat_len])?.try_into().unwrap();

        let encoded: ArrayD<f32> = encoded.view()?.into_owned();
        let batch = encoded.shape()[0];
        let max_frames = encoded.shape()[2];

        let mut decoder_stats = CallStats::default();
        let mut joint_stats = CallStats::default();

        const BEAM_SIZE: usize = 4;
        const DUR_BEAM_K: usize = 2;

        struct Beam {
            score: f32,
            tokens: Vec<usize>,
            last_frame: usize,
            dec_out: Value,
            state_0: Value,
            state_1: Value,
        }

        let init_token = Value::from_slice(&[1, 1], &[0i32])?;
        let init_s0: Value = Array3::<f32>::zeros([2, 1, 640]).try_into()?;
        let init_s1: Value = Array3::<f32>::zeros([2, 1, 640]).try_into()?;
        let t = Instant::now();
        let [dec_out, state_0, state_1] =
            self.decoder.run([init_token, init_s0, init_s1])?.try_into().unwrap();
        decoder_stats.record(batch, t.elapsed());

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

            let frame: Value =
                encoded.slice_axis(Axis(2), (frame_ix..frame_ix + 1).into()).try_into()?;

            while !hyps.is_empty() {
                let best_idx = hyps
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, b)| FloatOrd(b.score))
                    .map(|(i, _)| i)
                    .unwrap();
                let max_hyp = hyps.remove(best_idx);

                let t = Instant::now();
                let [logits] = self
                    .joint
                    .run([frame.clone(), max_hyp.dec_out.clone()])?
                    .try_into()
                    .unwrap();
                joint_stats.record(batch, t.elapsed());
                let logits = logits.view::<f32>()?;
                let logits = logits.as_slice().unwrap();

                let log_probs = log_softmax(&logits[0..=self.blank_id]);
                let dur_log_probs = log_softmax(&logits[self.blank_id + 1..]);

                // Non-blank expansions: top BEAM_SIZE tokens, duration fixed at 0
                let mut token_scores: Vec<(usize, f32)> =
                    (0..self.blank_id).map(|ti| (ti, log_probs[ti])).collect();
                token_scores.sort_by(|a, b| FloatOrd(b.1).cmp(&FloatOrd(a.1)));
                token_scores.truncate(BEAM_SIZE);

                let n = token_scores.len();

                // 1. Build batched inputs
                let token_ids: Vec<i32> =
                    token_scores.iter().map(|&(ti, _)| ti as i32).collect();
                let tokens_batch: Value =
                    Array2::<i32>::from_shape_fn((n, 1), |(i, _)| token_ids[i]).try_into()?;

                let s0_src = max_hyp.state_0.view::<f32>()?; // [2, 1, 640]
                let s0_batch: Value =
                    Array3::<f32>::from_shape_fn((2, n, 640), |(l, _, h)| s0_src[[l, 0, h]])
                        .try_into()?;

                let s1_src = max_hyp.state_1.view::<f32>()?;
                let s1_batch: Value =
                    Array3::<f32>::from_shape_fn((2, n, 640), |(l, _, h)| s1_src[[l, 0, h]])
                        .try_into()?;

                // 2. Single decoder call
                let t = Instant::now();
                let [dec_out_b, s0_b, s1_b] =
                    self.decoder.run([tokens_batch, s0_batch, s1_batch])?.try_into().unwrap();
                decoder_stats.record(n, t.elapsed());

                // 3. Slice outputs and push beams
                let dec_arr = dec_out_b.view::<f32>()?; // [n, hidden]
                let s0_arr = s0_b.view::<f32>()?; // [2, n, 640]
                let s1_arr = s1_b.view::<f32>()?; // [2, n, 640]

                for (i, &(ti, lp)) in token_scores.iter().enumerate() {
                    let new_dec_out: Value =
                        dec_arr.slice_axis(Axis(0), (i..i + 1).into()).try_into()?;
                    let new_s0: Value =
                        s0_arr.slice_axis(Axis(1), (i..i + 1).into()).try_into()?;
                    let new_s1: Value =
                        s1_arr.slice_axis(Axis(1), (i..i + 1).into()).try_into()?;
                    let mut new_tokens = max_hyp.tokens.clone();
                    new_tokens.push(ti);
                    hyps.push(Beam {
                        score: max_hyp.score + lp,
                        tokens: new_tokens,
                        last_frame: frame_ix,
                        dec_out: new_dec_out,
                        state_0: new_s0,
                        state_1: new_s1,
                    });
                }

                // Blank expansions: top DUR_BEAM_K non-zero durations
                let mut dur_scores: Vec<(usize, f32)> =
                    (1..dur_log_probs.len()).map(|di| (di, dur_log_probs[di])).collect();
                dur_scores.sort_by(|a, b| FloatOrd(b.1).cmp(&FloatOrd(a.1)));
                dur_scores.truncate(DUR_BEAM_K);

                for (di, dlp) in dur_scores {
                    kept.push(Beam {
                        score: max_hyp.score + log_probs[self.blank_id] + dlp,
                        tokens: max_hyp.tokens.clone(),
                        last_frame: frame_ix + di,
                        dec_out: max_hyp.dec_out.clone(),
                        state_0: max_hyp.state_0.clone(),
                        state_1: max_hyp.state_1.clone(),
                    });
                }

                // Prune combined pool to BEAM_SIZE
                let mut all: Vec<Beam> = hyps.drain(..).chain(kept.drain(..)).collect();
                all.sort_by(|a, b| FloatOrd(b.score).cmp(&FloatOrd(a.score)));
                all.truncate(BEAM_SIZE);
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
        Ok((best.tokens.into_iter().map(|t| self.vocab[t].as_str()).join(""), decoder_stats, joint_stats))
    }
}

fn main() -> anyhow::Result<()> {
    let nnef = tract_rs::nnef()?.with_tract_core()?.with_tract_transformers()?;
    let gpu = ["cuda", "metal", "default"]
        .iter()
        .find_map(|rt| tract_rs::runtime_for_name(rt).ok())
        .unwrap();

    let model = TdtModel::load(Path::new("assets/model"), &nnef, &gpu)?;

    let wav: Vec<f32> = hound::WavReader::open("assets/2086-149220-0033.wav")?
        .samples::<i16>()
        .map(|x| x.unwrap() as f32)
        .collect();

    model.transcribe_greedy(&wav)?;
    let (transcript_g, dec, joint) = model.transcribe_greedy(&wav)?;
    eprintln!("[greedy][decoder] {dec:?}");
    eprintln!("[greedy][joint]   {joint:?}");

    model.transcribe_beam(&wav)?;
    let (transcript_b, dec, joint) = model.transcribe_beam(&wav)?;
    eprintln!("[beam][decoder]   {dec:?}");
    eprintln!("[beam][joint]     {joint:?}");

    println!("Greedy: {transcript_g}");
    println!("Beam:   {transcript_b}");
    assert_eq!(
        transcript_g,
        "▁Well,▁I▁don't▁wish▁to▁see▁it▁any▁more,▁observed▁Phoebe,▁turning▁away▁her▁eyes."
    );
    assert_eq!(
        transcript_b,
        "▁Well,▁I▁don't▁wish▁to▁see▁it▁any▁more,▁observed▁Phoebe,▁turning▁away▁her▁eyes."
    );
    Ok(())
}
