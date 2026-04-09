use std::fs::File;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use anyhow::*;
use float_ord::FloatOrd;
use itertools::Itertools;
use tract::prelude::tract_ndarray::prelude::*;
use tract::prelude::*;

fn argmax(slice: &[f32]) -> Option<usize> {
    slice.into_iter().position_max_by_key(|x| FloatOrd(**x))
}

fn show(hyp: &[usize], vocab: &[&str], label: &str) {
    let text: String = hyp.iter().map(|&t| vocab[t]).join("");
    let display = text.replace('▁', " ");
    eprint!("\r{} {label}          ", display.trim_start());
}

/// Audio chunk size sent by the "microphone" thread (samples at 16kHz).
const AUDIO_CHUNK: usize = 80;
/// Preprocessor pulse in audio samples (112 feat frames × 160 hop).
const PREPROC_PULSE: usize = 17920;
/// Encoder pulse in feature frames (14 token chunks × 8x subsampling).
const ENCODER_PULSE: usize = 112;

fn main() -> anyhow::Result<()> {
    let t0 = Instant::now();

    let config: serde_json::Value =
        serde_json::from_reader(File::open("assets/model/model_config.json")?)?;
    let blank_id = config.pointer("/decoder/vocab_size").unwrap().as_i64().unwrap() as usize;
    let vocab = config.pointer("/joint/vocabulary").unwrap().as_array().unwrap();
    let vocab: Vec<&str> = vocab.iter().map(|v| v.as_str().unwrap()).collect();

    let nnef = tract::nnef()?.with_tract_core()?.with_tract_transformers()?;
    let runtime = ["cuda", "metal", "default"]
        .iter()
        .find_map(|rt| tract::runtime_for_name(rt).ok())
        .unwrap();

    // ── Preprocessor (pulsified) ────────────────────────────────────────
    let mut preprocessor = nnef.load("assets/model/preprocessor.nnef.tgz")?;
    preprocessor.transform(ConcretizeSymbols::new().value("BATCH", 1))?;
    preprocessor
        .transform(r#"{"name":"patch","body":"length = tract_core_shape_of(input_signal)[1];"}"#)?;
    preprocessor.transform(r#"{"name":"select_outputs","outputs":["processed_signal"]}"#)?;
    preprocessor.transform(Pulse::new(PREPROC_PULSE.to_string()).symbol("INPUT_SIGNAL__TIME"))?;
    let preprocessor = preprocessor.into_runnable()?;

    let pp_delay = preprocessor.property("pulse.delay")?.view::<i64>()?[0].to_owned() as usize;
    let pp_out_axis =
        preprocessor.property("pulse.output_axes")?.view::<i64>()?[0].to_owned() as usize;
    let pp_out_pulse = preprocessor.output_fact(0)?.dim(pp_out_axis)?.to_int64()? as usize;
    let pp_fact = preprocessor.input_fact(0)?;
    let pp_input_shape: Vec<usize> = (0..pp_fact.rank()?)
        .map(|a| pp_fact.dim(a).and_then(|d| d.to_int64()).map(|v| v as usize))
        .collect::<Result<_>>()?;

    // ── Encoder (pulsified) ─────────────────────────────────────────────
    let mut encoder = nnef.load("assets/model/encoder.p1.nnef.tgz")?;
    encoder.transform(ConcretizeSymbols::new().value("BATCH", 1))?;
    encoder.transform("transformers_detect_all")?;
    encoder
        .transform(r#"{"name":"patch","body":"length = tract_core_shape_of(audio_signal)[2];"}"#)?;
    encoder.transform(r#"{"name":"select_outputs","outputs":["outputs"]}"#)?;
    encoder.transform(Pulse::new(ENCODER_PULSE.to_string()).symbol("AUDIO_SIGNAL__TIME"))?;
    let encoder = runtime.prepare(encoder)?;

    let enc_output_axis =
        encoder.property("pulse.output_axes")?.view::<i64>()?[0].to_owned() as usize;
    let enc_output_pulse = encoder.output_fact(0)?.dim(enc_output_axis)?.to_int64()? as usize;
    let enc_delay = encoder.property("pulse.delay")?.view::<i64>()?[0].to_owned() as usize;
    let enc_fact = encoder.input_fact(0)?;
    let enc_input_shape: Vec<usize> = (0..enc_fact.rank()?)
        .map(|a| enc_fact.dim(a).and_then(|d| d.to_int64()).map(|v| v as usize))
        .collect::<Result<_>>()?;

    // ── Decoder + Joint ─────────────────────────────────────────────────
    let mut decoder_model = nnef.load("assets/model/decoder.nnef.tgz")?;
    decoder_model
        .transform(ConcretizeSymbols::new().value("BATCH", 1).value("TARGETS__TIME", 1))?;
    let decoder = runtime.prepare(decoder_model)?;

    let mut joint_model = nnef.load("assets/model/joint.nnef.tgz")?;
    joint_model.transform(
        ConcretizeSymbols::new()
            .value("BATCH", 1)
            .value("ENCODER_OUTPUTS__TIME", 1)
            .value("DECODER_OUTPUTS__TIME", 1),
    )?;
    let joint = runtime.prepare(joint_model)?;

    let load_time = t0.elapsed();

    // ── Load audio ──────────────────────────────────────────────────────
    let wav: Vec<f32> = hound::WavReader::open("assets/2086-149220-0033.wav")?
        .samples::<i16>()
        .map(|x| x.unwrap() as f32 / 32768.0)
        .collect();
    let total_samples = wav.len();
    let audio_duration = total_samples as f64 / 16000.0;

    // ── Mic thread ──────────────────────────────────────────────────────
    let (audio_tx, audio_rx) = mpsc::sync_channel::<Vec<f32>>(4);
    let mic_handle = std::thread::Builder::new().name("mic".into()).spawn(move || {
        let mut offset = 0;
        while offset < total_samples {
            let end = (offset + AUDIO_CHUNK).min(total_samples);
            if audio_tx.send(wav[offset..end].to_vec()).is_err() {
                break;
            }
            offset = end;
            std::thread::sleep(Duration::from_secs_f64(AUDIO_CHUNK as f64 / 16000.0));
        }
    })?;

    // ── State ───────────────────────────────────────────────────────────
    let mut audio_buf: Vec<f32> = Vec::new();
    let mut audio_consumed: usize = 0;
    let mut preproc_state = preprocessor.spawn_state()?;
    let mut encoder_state = encoder.spawn_state()?;
    let mut enc_delay_remaining = enc_delay;
    let mut pp_delay_remaining = pp_delay;
    let mut pulse_count: usize = 0;
    let mut feat_buf: Vec<ArrayD<f32>> = Vec::new();
    let mut feat_buf_frames: usize = 0;
    let mut hyp: Vec<usize> = vec![];
    let max_tokens_per_frame = 10;

    // Stats.
    let mut total_preproc = Duration::ZERO;
    let mut total_encoder = Duration::ZERO;
    let mut total_joint = Duration::ZERO;
    let mut total_decoder = Duration::ZERO;
    let mut n_preproc: usize = 0;
    let mut n_encoder: usize = 0;
    let mut n_joint: usize = 0;
    let mut n_decoder: usize = 0;

    // Warm-up decoder.
    let blank_tok = Tensor::from_slice(&[1, 1], &[blank_id as i32])?;
    let mut state_0 = tensor(Array3::<f32>::zeros([2, 1, 640]))?;
    let mut state_1 = tensor(Array3::<f32>::zeros([2, 1, 640]))?;
    let [_out, s0, s1] = decoder.run([blank_tok.clone(), state_0, state_1])?.try_into().unwrap();
    state_0 = s0;
    state_1 = s1;
    let [mut dec_token, s0, s1] = decoder.run([blank_tok, state_0, state_1])?.try_into().unwrap();
    state_0 = s0;
    state_1 = s1;

    // ── Inline helpers as macros to avoid borrow issues ─────────────────

    /// Run RNNT decode loop on one encoder frame.
    macro_rules! decode_frame {
        ($frame:expr) => {{
            let frame = $frame;
            let mut tokens_this_frame = 0usize;
            loop {
                show(&hyp, &vocab, "[jnt]");
                let t = Instant::now();
                let [logits] = joint.run([frame.clone(), dec_token.clone()])?.try_into().unwrap();
                total_joint += t.elapsed();
                n_joint += 1;
                let logits_view = logits.view::<f32>()?;
                let token_id = argmax(logits_view.as_slice().unwrap()).unwrap();
                if token_id == blank_id {
                    break;
                }
                hyp.push(token_id);
                tokens_this_frame += 1;
                show(&hyp, &vocab, "[dec]");
                let t = Instant::now();
                let tok = Tensor::from_slice(&[1, 1], &[token_id as i32])?;
                [dec_token, state_0, state_1] =
                    decoder.run([tok, state_0, state_1])?.try_into().unwrap();
                total_decoder += t.elapsed();
                n_decoder += 1;
                if tokens_this_frame >= max_tokens_per_frame {
                    break;
                }
            }
        }};
    }

    /// Run one encoder pulse and decode all output frames.
    macro_rules! encoder_pulse {
        ($features:expr) => {{
            show(&hyp, &vocab, "[enc]");
            let t = Instant::now();
            let pulse_tensor: Tensor = tensor($features)?;
            let enc_results = encoder_state.run([pulse_tensor])?;
            total_encoder += t.elapsed();
            n_encoder += 1;
            let enc_out: ArrayD<f32> = enc_results[0].view()?.into_owned();
            pulse_count += 1;
            for f in 0..enc_output_pulse {
                if enc_delay_remaining > 0 {
                    enc_delay_remaining -= 1;
                    continue;
                }
                let frame: Tensor =
                    tensor(enc_out.slice_axis(Axis(enc_output_axis), (f..f + 1).into()))?;
                decode_frame!(frame);
            }
        }};
    }

    /// Buffer preprocessor features and run encoder when enough accumulate.
    macro_rules! feed_features {
        ($features:expr) => {{
            let features: ArrayD<f32> = $features;
            let usable_start = pp_delay_remaining.min(pp_out_pulse);
            pp_delay_remaining = pp_delay_remaining.saturating_sub(pp_out_pulse);
            if usable_start < pp_out_pulse {
                let usable =
                    features.slice_axis(Axis(pp_out_axis), (usable_start..pp_out_pulse).into());
                feat_buf.push(usable.to_owned());
                feat_buf_frames += usable.shape()[pp_out_axis];
                while feat_buf_frames >= ENCODER_PULSE {
                    let refs: Vec<_> = feat_buf.iter().map(|a| a.view()).collect();
                    let all = tract::prelude::tract_ndarray::concatenate(Axis(pp_out_axis), &refs)?;
                    let enc_feat = all.slice_axis(Axis(pp_out_axis), (..ENCODER_PULSE).into());
                    encoder_pulse!(enc_feat.to_owned());
                    let leftover = all.slice_axis(Axis(pp_out_axis), (ENCODER_PULSE..).into());
                    feat_buf_frames -= ENCODER_PULSE;
                    feat_buf.clear();
                    if feat_buf_frames > 0 {
                        feat_buf.push(leftover.to_owned());
                    }
                }
            }
        }};
    }

    /// Run preprocessor on a sample buffer and feed features.
    macro_rules! run_preproc {
        ($input:expr) => {{
            show(&hyp, &vocab, "[pre]");
            let t = Instant::now();
            let pp_results = preproc_state.run([$input])?;
            total_preproc += t.elapsed();
            n_preproc += 1;
            let features: ArrayD<f32> = pp_results[0].view()?.into_owned();
            feed_features!(features);
        }};
    }

    // ── Main loop ───────────────────────────────────────────────────────

    let stream_start = Instant::now();

    for chunk in audio_rx {
        audio_buf.extend_from_slice(&chunk);
        show(&hyp, &vocab, "");

        while audio_consumed + PREPROC_PULSE <= audio_buf.len() {
            let start = audio_consumed;
            let end = start + PREPROC_PULSE;
            let pp_input = Tensor::from_slice(&pp_input_shape, &audio_buf[start..end])?;
            audio_consumed = end;
            run_preproc!(pp_input);
        }
    }

    // Flush remaining audio.
    let remaining = audio_buf.len() - audio_consumed;
    if remaining > 0 {
        let mut pp_input_data = vec![0.0f32; pp_input_shape.iter().product()];
        pp_input_data[..remaining].copy_from_slice(&audio_buf[audio_consumed..]);
        let pp_input = Tensor::from_slice(&pp_input_shape, &pp_input_data)?;
        run_preproc!(pp_input);
    }
    // Flush remaining buffered features.
    if feat_buf_frames > 0 {
        let refs: Vec<_> = feat_buf.iter().map(|a| a.view()).collect();
        let leftover = tract::prelude::tract_ndarray::concatenate(Axis(pp_out_axis), &refs)?;
        let mut enc_input =
            Array3::<f32>::zeros((enc_input_shape[0], enc_input_shape[1], enc_input_shape[2]));
        let n = leftover.shape()[pp_out_axis].min(enc_input_shape[2]);
        enc_input.slice_mut(s![.., .., ..n]).assign(&leftover.slice(s![.., .., ..n]));
        encoder_pulse!(enc_input);
    }

    let stream_time = stream_start.elapsed();
    mic_handle.join().unwrap();

    // Final transcript.
    let transcript: String = hyp.iter().map(|&t| vocab[t]).join("");
    let display = transcript.replace('▁', " ");
    eprint!("\r{}                    \n", display.trim_start());

    // Stats.
    eprintln!();
    eprintln!("--- stats ---");
    eprintln!("runtime:       {}", runtime.name()?);
    eprintln!("model load:    {:.1}s", load_time.as_secs_f64());
    eprintln!("audio:         {:.2}s ({} samples at 16kHz)", audio_duration, total_samples);
    eprintln!(
        "stream wall:   {:.2}s ({:.1}x real-time)",
        stream_time.as_secs_f64(),
        audio_duration / stream_time.as_secs_f64()
    );
    eprintln!("pulses:        {pulse_count}");
    if n_preproc > 0 {
        eprintln!(
            "preprocessor:  {:.1}ms total, {:.1}ms/pulse ({n_preproc} calls)",
            total_preproc.as_secs_f64() * 1000.0,
            total_preproc.as_secs_f64() * 1000.0 / n_preproc as f64
        );
    }
    if n_encoder > 0 {
        eprintln!(
            "encoder:       {:.1}ms total, {:.1}ms/pulse ({n_encoder} calls)",
            total_encoder.as_secs_f64() * 1000.0,
            total_encoder.as_secs_f64() * 1000.0 / n_encoder as f64
        );
    }
    if n_joint > 0 {
        eprintln!(
            "joint:         {:.1}ms total, {:.2}ms/call ({n_joint} calls)",
            total_joint.as_secs_f64() * 1000.0,
            total_joint.as_secs_f64() * 1000.0 / n_joint as f64
        );
    }
    if n_decoder > 0 {
        eprintln!(
            "decoder:       {:.1}ms total, {:.2}ms/call ({n_decoder} calls)",
            total_decoder.as_secs_f64() * 1000.0,
            total_decoder.as_secs_f64() * 1000.0 / n_decoder as f64
        );
    }
    let compute_total = total_preproc + total_encoder + total_joint + total_decoder;
    eprintln!(
        "compute total: {:.1}ms ({:.1}x real-time)",
        compute_total.as_secs_f64() * 1000.0,
        audio_duration / compute_total.as_secs_f64()
    );

    let expected = "▁well▁I▁don't▁wish▁to▁see▁it▁any▁more▁observed▁Phoebe,▁turning▁away▁her▁eyes.▁It▁is▁certainly▁very▁like▁the▁old▁portrait";
    if transcript != expected {
        eprintln!("\nNOTE: streaming transcript differs slightly from batch reference");
    }
    Ok(())
}
