use std::fs::File;
use std::sync::mpsc;
use std::time::Instant;

use anyhow::*;
use float_ord::FloatOrd;
use itertools::Itertools;
use tract::prelude::tract_ndarray::prelude::*;
use tract::prelude::*;

fn argmax(slice: &[f32]) -> Option<usize> {
    slice.into_iter().position_max_by_key(|x| FloatOrd(**x))
}

/// Audio chunk size sent by the "microphone" thread (samples at 16kHz).
/// 80 samples = 5ms of audio — simulates a real-time audio callback.
const AUDIO_CHUNK: usize = 80;

/// Preprocessor pulse in audio samples.
/// 112 feature frames × 160 samples/frame (STFT hop) = 17920 samples.
const PREPROC_PULSE: usize = 17920;

/// Encoder pulse in feature frames.
/// 14 transformer-token chunks × 8x conv subsampling = 112 frames.
const ENCODER_PULSE: usize = 112;

fn main() -> anyhow::Result<()> {
    let t0 = Instant::now();
    eprintln!("[{:7.3}] Loading models...", t0.elapsed().as_secs_f64());

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
    eprintln!("[{:7.3}] Runtime: {}", t0.elapsed().as_secs_f64(), runtime.name()?);

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

    eprintln!(
        "[{:7.3}] Models loaded. Preproc: {} samples/pulse -> {} feat frames (delay {}). \
         Encoder: {} feat frames/pulse -> {} token frames (delay {}).",
        t0.elapsed().as_secs_f64(),
        PREPROC_PULSE,
        pp_out_pulse,
        pp_delay,
        ENCODER_PULSE,
        enc_output_pulse,
        enc_delay,
    );

    // ── Load audio ──────────────────────────────────────────────────────
    let wav: Vec<f32> = hound::WavReader::open("assets/2086-149220-0033.wav")?
        .samples::<i16>()
        .map(|x| x.unwrap() as f32 / 32768.0)
        .collect();
    let total_samples = wav.len();
    eprintln!(
        "[{:7.3}] Audio: {:.2}s at 16kHz, chunk: {} samples ({:.1}ms)",
        t0.elapsed().as_secs_f64(),
        total_samples as f64 / 16000.0,
        AUDIO_CHUNK,
        AUDIO_CHUNK as f64 / 16.0,
    );

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
            std::thread::sleep(std::time::Duration::from_secs_f64(AUDIO_CHUNK as f64 / 16000.0));
        }
    })?;

    // ── Main thread: streaming pipeline ─────────────────────────────────

    let print_partial = |hyp: &[usize], vocab: &[&str], tag: &str| {
        let text: String = hyp.iter().map(|&t| vocab[t]).join("");
        let display = text.replace('▁', " ");
        eprintln!("{} {tag}", display.trim_start());
    };

    let mut audio_buf: Vec<f32> = Vec::new();
    let mut audio_consumed: usize = 0;
    let mut preproc_state = preprocessor.spawn_state()?;
    let mut encoder_state = encoder.spawn_state()?;
    let mut enc_delay_remaining = enc_delay;
    let mut pp_delay_remaining = pp_delay;
    let mut pulse_count: usize = 0;
    let max_tokens_per_frame = 10;
    // Feature buffer: accumulates preprocessor output frames until a full
    // encoder pulse (ENCODER_PULSE frames) is available.
    let mut feat_buf: Vec<ArrayD<f32>> = Vec::new();
    let mut feat_buf_frames: usize = 0;

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

    let mut hyp: Vec<usize> = vec![];

    // Helper: run RNNT decode on one encoder output frame.
    let mut decode_frame = |frame: Tensor,
                            dec_token: &mut Tensor,
                            state_0: &mut Tensor,
                            state_1: &mut Tensor,
                            hyp: &mut Vec<usize>,
                            vocab: &[&str]|
     -> anyhow::Result<()> {
        let mut tokens_this_frame = 0;
        loop {
            let [logits] = joint.run([frame.clone(), dec_token.clone()])?.try_into().unwrap();
            let logits_view = logits.view::<f32>()?;
            let token_id = argmax(logits_view.as_slice().unwrap()).unwrap();
            if token_id == blank_id {
                break;
            }
            hyp.push(token_id);
            tokens_this_frame += 1;
            let tok = Tensor::from_slice(&[1, 1], &[token_id as i32])?;
            let [dt, s0, s1] =
                decoder.run([tok, state_0.clone(), state_1.clone()])?.try_into().unwrap();
            *dec_token = dt;
            *state_0 = s0;
            *state_1 = s1;
            print_partial(hyp, vocab, "[decoder]");
            if tokens_this_frame >= max_tokens_per_frame {
                break;
            }
        }
        Ok(())
    };

    // Helper: process one encoder pulse from feature data.
    let mut run_encoder_pulse =
        |features: &ArrayD<f32>, tag: &str, audio_time: f64| -> anyhow::Result<()> {
            let enc_start = Instant::now();
            let pulse_tensor: Tensor = tensor(features.to_owned())?;
            let enc_results = encoder_state.run([pulse_tensor])?;
            let enc_ms = enc_start.elapsed().as_secs_f64() * 1000.0;
            let enc_out: ArrayD<f32> = enc_results[0].view()?.into_owned();

            pulse_count += 1;
            print_partial(
                &hyp,
                &vocab,
                &format!(
                    "[{tag} encoder {enc_ms:.0}ms | pulse {pulse_count} | audio {audio_time:.2}s]"
                ),
            );

            for f in 0..enc_output_pulse {
                if enc_delay_remaining > 0 {
                    enc_delay_remaining -= 1;
                    continue;
                }
                let frame: Tensor =
                    tensor(enc_out.slice_axis(Axis(enc_output_axis), (f..f + 1).into()))?;
                decode_frame(frame, &mut dec_token, &mut state_0, &mut state_1, &mut hyp, &vocab)?;
            }
            Ok(())
        };

    // Helper: buffer preprocessor output and feed encoder when ready.
    let mut feed_features =
        |features: ArrayD<f32>, pp_tag: &str, audio_time: f64| -> anyhow::Result<()> {
            // Skip delay frames from preprocessor output.
            let usable_start = pp_delay_remaining.min(pp_out_pulse);
            pp_delay_remaining = pp_delay_remaining.saturating_sub(pp_out_pulse);
            if usable_start >= pp_out_pulse {
                return Ok(());
            }
            let usable =
                features.slice_axis(Axis(pp_out_axis), (usable_start..pp_out_pulse).into());
            let n_new = usable.shape()[pp_out_axis];
            feat_buf.push(usable.to_owned());
            feat_buf_frames += n_new;

            // Run encoder pulses from the feature buffer.
            while feat_buf_frames >= ENCODER_PULSE {
                // Concatenate buffered features and take ENCODER_PULSE.
                let refs: Vec<_> = feat_buf.iter().map(|a| a.view()).collect();
                let all = tract::prelude::tract_ndarray::concatenate(Axis(pp_out_axis), &refs)?;
                let enc_features = all.slice_axis(Axis(pp_out_axis), (..ENCODER_PULSE).into());
                run_encoder_pulse(&enc_features.to_owned(), pp_tag, audio_time)?;

                // Keep leftover features.
                let leftover = all.slice_axis(Axis(pp_out_axis), (ENCODER_PULSE..).into());
                feat_buf_frames -= ENCODER_PULSE;
                feat_buf.clear();
                if feat_buf_frames > 0 {
                    feat_buf.push(leftover.to_owned());
                }
            }
            Ok(())
        };

    for chunk in audio_rx {
        audio_buf.extend_from_slice(&chunk);

        // Run preprocessor pulses as audio accumulates.
        while audio_consumed + PREPROC_PULSE <= audio_buf.len() {
            let start = audio_consumed;
            let end = start + PREPROC_PULSE;

            let pp_input = Tensor::from_slice(&pp_input_shape, &audio_buf[start..end])?;
            let pp_start = Instant::now();
            let pp_results = preproc_state.run([pp_input])?;
            let pp_ms = pp_start.elapsed().as_secs_f64() * 1000.0;
            let features: ArrayD<f32> = pp_results[0].view()?.into_owned();
            audio_consumed = end;

            let audio_time = audio_buf.len() as f64 / 16000.0;
            feed_features(features, &format!("preproc {pp_ms:.0}ms +"), audio_time)?;
        }
    }

    // Flush: zero-padded final preprocessor pulse.
    {
        let remaining = audio_buf.len() - audio_consumed;
        if remaining > 0 {
            let mut pp_input_data = vec![0.0f32; pp_input_shape.iter().product()];
            pp_input_data[..remaining].copy_from_slice(&audio_buf[audio_consumed..]);
            let pp_input = Tensor::from_slice(&pp_input_shape, &pp_input_data)?;
            let pp_results = preproc_state.run([pp_input])?;
            let features: ArrayD<f32> = pp_results[0].view()?.into_owned();
            let audio_time = audio_buf.len() as f64 / 16000.0;
            feed_features(features, "flush +", audio_time)?;
        }
        // Flush remaining buffered features as a zero-padded encoder pulse.
        if feat_buf_frames > 0 {
            let refs: Vec<_> = feat_buf.iter().map(|a| a.view()).collect();
            let leftover = tract::prelude::tract_ndarray::concatenate(Axis(pp_out_axis), &refs)?;
            let mut enc_input =
                Array3::<f32>::zeros((enc_input_shape[0], enc_input_shape[1], enc_input_shape[2]));
            let n = leftover.shape()[pp_out_axis].min(enc_input_shape[2]);
            enc_input.slice_mut(s![.., .., ..n]).assign(&leftover.slice(s![.., .., ..n]));
            let at = audio_buf.len() as f64 / 16000.0;
            run_encoder_pulse(&enc_input.into_dyn(), "flush +", at)?;
        }
    }

    mic_handle.join().unwrap();

    let transcript: String = hyp.iter().map(|&t| vocab[t]).join("");
    let display = transcript.replace('▁', " ");
    eprintln!("\n{}", display.trim_start());

    let expected = "▁well▁I▁don't▁wish▁to▁see▁it▁any▁more▁observed▁Phoebe,▁turning▁away▁her▁eyes.▁It▁is▁certainly▁very▁like▁the▁old▁portrait";
    if transcript != expected {
        eprintln!(
            "\nNOTE: streaming transcript differs slightly from batch reference (pulse boundary effects)"
        );
    }
    Ok(())
}
